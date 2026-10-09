"""Turn a :class:`~src.services.curation.vlm_policy.VlmPolicy` into query clauses.

The one place both VLM selectors (the continuous worker and the auto-label
sweep) get their scope clauses from, so a policy selects the same crops through
either. The selectors keep their own eligibility rules (validated, classifier
confident, already answered, ...); the clauses here only narrow that set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.curation import items_index
from src.services.curation.cluster_representatives import representatives_msearch_body
from src.services.curation.vlm_class_attempt import VLM_CLASS_ATTEMPTED_AT_FIELD
from src.services.curation.vlm_policy_store import get_vlm_policy


if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from src.services.curation.vlm_policy import VlmPolicy


# Stable per-crop selection: the same crop gets the same answer on every poll,
# so a sampled-out crop is not picked up later by chance.
_SAMPLE_SOURCE = (
    "doc['crop_id'].size() > 0 && "
    "(Math.abs(doc['crop_id'].value.hashCode() % 10000) / 10000.0) < params.frac"
)

_MAX_CLUSTERS = 5000
_MSEARCH_CHUNK = 200


@dataclass(frozen=True)
class ScopeClauses:
    """``filter`` and ``must_not`` clauses to add to a selector's bool query."""

    filter: list[dict[str, Any]] = field(default_factory=list)
    must_not: list[dict[str, Any]] = field(default_factory=list)


MATCH_NONE: dict[str, Any] = {'match_none': {}}


def vlm_scope_clauses(
    policy: VlmPolicy, *, representative_ids: Collection[str] | None
) -> ScopeClauses:
    """The clauses ``policy`` adds. ``representatives`` needs the resolved
    representative crop ids (:func:`representative_ids`); without them the
    selector cannot tell which crops qualify, so it refuses rather than
    falling back to every crop."""
    filters: list[dict[str, Any]] = []
    if policy.scope == 'off':
        filters.append(MATCH_NONE)
    elif policy.scope == 'uncertain':
        filters.append(
            {
                'bool': {
                    'should': [
                        {'range': {'confidence': {'lt': policy.conf_max}}},
                        {'bool': {'must_not': [{'exists': {'field': 'confidence'}}]}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
    elif policy.scope == 'representatives':
        if representative_ids is None:
            msg = 'scope=representatives needs the resolved representative ids'
            raise ValueError(msg)
        filters.append(
            {
                'bool': {
                    'should': [
                        {'terms': {'crop_id': sorted(representative_ids)}},
                        {'range': {'cluster_id': {'lt': 0}}},
                        {'bool': {'must_not': [{'exists': {'field': 'cluster_id'}}]}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
    if policy.sample_frac < 1.0:
        filters.append(
            {
                'script': {
                    'script': {
                        'lang': 'painless',
                        'source': _SAMPLE_SOURCE,
                        'params': {'frac': policy.sample_frac},
                    }
                }
            }
        )
    return ScopeClauses(filter=filters)


async def representative_ids(opensearch: Any, *, per_cluster: int) -> set[str]:
    """The ``per_cluster`` crops nearest each assigned cluster's centre (the same
    ranking the cluster cards show), excluded items left out. Validated crops
    count toward the K so an already-labelled representative is not replaced by
    the next-nearest one."""
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': {
                'bool': {
                    'filter': [{'range': {'cluster_id': {'gte': 0}}}],
                    'must_not': [{'term': {'class_excluded': True}}],
                }
            },
            'aggs': {'clusters': {'terms': {'field': 'cluster_id', 'size': _MAX_CLUSTERS}}},
        },
    )
    cluster_ids = [
        int(b['key']) for b in resp.get('aggregations', {}).get('clusters', {}).get('buckets', [])
    ]
    ids: set[str] = set()
    for start in range(0, len(cluster_ids), _MSEARCH_CHUNK):
        lines: list[dict[str, Any]] = []
        for cid in cluster_ids[start : start + _MSEARCH_CHUNK]:
            lines.append({'index': items_index()})
            lines.append(representatives_msearch_body(cid, per_cluster))
        out = await opensearch.msearch(body=lines)
        for sub in out.get('responses', []):
            for hit in ((sub or {}).get('hits') or {}).get('hits') or []:
                crop_id = (hit.get('_source') or {}).get('crop_id') or hit.get('_id')
                if crop_id:
                    ids.add(crop_id)
    return ids


class ScopeCache:
    """A project's policy and representative ids for ``ttl_s`` seconds, so a
    polling worker neither re-reads the settings document nor re-ranks every
    cluster on each poll. Keyed by project slug; call under that project's binding."""

    def __init__(self, *, ttl_s: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._entries: dict[str, tuple[float, VlmPolicy, set[str] | None]] = {}

    async def get(self, opensearch: Any, slug: str) -> tuple[VlmPolicy, set[str] | None]:
        hit = self._entries.get(slug)
        if hit is not None and self._clock() - hit[0] < self._ttl_s:
            return hit[1], hit[2]
        policy = await get_vlm_policy(opensearch)
        reps = (
            await representative_ids(opensearch, per_cluster=policy.per_cluster)
            if policy.scope == 'representatives'
            else None
        )
        self._entries[slug] = (self._clock(), policy, reps)
        return policy, reps


async def daily_budget_remaining(
    opensearch: Any, policy: VlmPolicy, *, now: datetime | None = None
) -> int | None:
    """VLM class attempts still allowed today (UTC), or ``None`` when the policy
    sets no daily cap. Counts every attempt, answered or empty."""
    if policy.max_crops_per_day <= 0:
        return None
    moment = now or datetime.now(UTC)
    midnight = moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    used = (
        await opensearch.count(
            index=items_index(),
            body={
                'query': {'range': {VLM_CLASS_ATTEMPTED_AT_FIELD: {'gte': midnight.isoformat()}}}
            },
        )
    ).get('count', 0)
    return max(0, policy.max_crops_per_day - int(used))

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

from src.config.curation_accessors import items_index
from src.config.vlm_class_attempt import VLM_CLASS_ATTEMPTED_AT_FIELD
from src.services.curation.cluster_representatives import partition_representatives_body
from src.services.curation.policy_doc_store import PolicyConflictError
from src.services.curation.vlm_policy_store import get_vlm_policy
from src.services.curation.vlm_rep_claims import (
    RepClaim,
    RepClaims,
    read_rep_claims,
    write_rep_claims,
)


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
_CLAIM_WRITE_ATTEMPTS = 3


@dataclass(frozen=True)
class ScopeClauses:
    """``filter`` and ``must_not`` clauses to add to a selector's bool query."""

    filter: list[dict[str, Any]] = field(default_factory=list)
    must_not: list[dict[str, Any]] = field(default_factory=list)


MATCH_NONE: dict[str, Any] = {'match_none': {}}

# Never sent to the VLM by the global sweep or the worker, whatever the scope: the frozen
# test holdout and excluded items are out of the labelling pipeline.
OUT_OF_PIPELINE: tuple[dict[str, Any], ...] = (
    {'term': {'test_holdout': True}},
    {'term': {'class_excluded': True}},
)


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
                        {'term': {'cluster_state': 'unassigned'}},
                        {'bool': {'must_not': [{'exists': {'field': 'cluster_uid'}}]}},
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
    return ScopeClauses(filter=filters, must_not=list(OUT_OF_PIPELINE))


async def _ranked_by_cluster(opensearch: Any, per_cluster: int) -> dict[str, list[str]]:
    """Each served cluster's ``per_cluster`` best-scored members, best first (the ranking the
    cluster cards show), excluded items left out."""
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': {
                'bool': {
                    'filter': [{'exists': {'field': 'cluster_uid'}}],
                    'must_not': [
                        {'term': {'class_excluded': True}},
                        {'term': {'cluster_state': 'unassigned'}},
                    ],
                }
            },
            'aggs': {'clusters': {'terms': {'field': 'cluster_uid', 'size': _MAX_CLUSTERS}}},
        },
    )
    uids = [
        str(b['key']) for b in resp.get('aggregations', {}).get('clusters', {}).get('buckets', [])
    ]
    ranked: dict[str, list[str]] = {}
    for start in range(0, len(uids), _MSEARCH_CHUNK):
        chunk = uids[start : start + _MSEARCH_CHUNK]
        lines: list[dict[str, Any]] = []
        for uid in chunk:
            lines.append({'index': items_index()})
            lines.append(partition_representatives_body(uid, per_cluster))
        out = await opensearch.msearch(body=lines)
        for uid, sub in zip(chunk, out.get('responses', []), strict=True):
            hits = ((sub or {}).get('hits') or {}).get('hits') or []
            ranked[uid] = [h.get('_source', {}).get('crop_id') or h['_id'] for h in hits]
    return ranked


def _reconcile_claims(
    stored: RepClaims, ranked: dict[str, list[str]], per_cluster: int
) -> RepClaims:
    """``stored`` brought up to date with the served clusters.

    A cluster keeps the reps it claimed (topped up to ``per_cluster`` from its current ranking
    when it claimed fewer); only a cluster with no claim is ranked afresh, so labelling a rep
    never opens the pool for the next ones. A ``cluster_uid`` is never reused, so a claim needs
    no staleness check; claims of clusters that no longer exist are dropped.
    """
    by_cluster = {c.cluster_uid: c.crop_ids for c in stored.claims}
    claims: list[RepClaim] = []
    for uid, fresh in ranked.items():
        ids = by_cluster.get(uid)
        if ids is None:
            ids = list(fresh)
        elif len(ids) < per_cluster:
            ids = [*ids, *(i for i in fresh if i not in ids)][:per_cluster]
        claims.append(RepClaim(cluster_uid=uid, crop_ids=ids))
    return RepClaims(revision=stored.revision, claims=claims)


async def representative_ids(opensearch: Any, *, per_cluster: int) -> set[str]:
    """The crops the ``representatives`` scope may send to the VLM: each assigned
    cluster's ``per_cluster`` claimed reps (see :mod:`vlm_rep_claims`), excluded
    items left out. Claims are persisted, so the set is stable across refreshes
    and restarts; a concurrent writer is re-read, never overwritten."""
    ranked = await _ranked_by_cluster(opensearch, per_cluster)
    for _ in range(_CLAIM_WRITE_ATTEMPTS):
        stored, row_revision = await read_rep_claims()
        updated = _reconcile_claims(stored, ranked, per_cluster)
        if updated.claims != stored.claims:
            try:
                await write_rep_claims(
                    updated.model_copy(update={'revision': stored.revision + 1}), row_revision
                )
            except PolicyConflictError:
                continue
        return {i for c in updated.claims for i in c.crop_ids[:per_cluster]}
    msg = 'could not store the VLM representative claims (concurrent writers)'
    raise RuntimeError(msg)


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
        policy = await get_vlm_policy()
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

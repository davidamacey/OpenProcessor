"""Region-profile activation impact aggregation (W4, any_domain_plan.md
§4.6): "if I activate this profile, what happens to existing items",
served by ``GET /region_profiles/active/impact`` and embedded in the
activate response.

No silent data rewrite, ever (§4.6): this module only counts; nothing
here writes to an item.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.config import PENDING_STATUSES, RegionStatus
from src.config.curation import items_index
from src.config.region_fields import get_region_fields
from src.services.curation.region_eval import LEGACY_STATUS_ALIASES
from src.services.curation.region_requeue import REQUEUEABLE_STATUSES
from src.services.curation.region_scope import parent_classes_clause
from src.services.curation.reprocess_models import (
    ReprocessFilter,
    ReprocessRequest,
    ReprocessTargets,
)
from src.services.curation.reprocess_targets import selector_clauses


if TYPE_CHECKING:
    from src.config import DetectionProfile

#: Both the current and legacy pending-status spellings (operators rename
#: 'pending' -> 'pending_detection' etc.; the worker reads both). Sourced
#: from the enum module + region_eval's alias map, not re-literaled here.
_PENDING_STATUSES: tuple[str, ...] = tuple(status.value for status in PENDING_STATUSES) + tuple(
    LEGACY_STATUS_ALIASES
)

#: Sentinel terms-agg bucket key for "no region_profile stamp at all"
#: (an item ingested while region detection was off, or never in scope).
_UNSEEDED_KEY = '__unseeded__'


class ActivationImpactByProfile(BaseModel):
    name: str | None
    revision: int | None
    count: int


class ActivationImpact(BaseModel):
    """§4.6/§7.3.

    ``suggested_reprocess`` is the explicit re-run for "every unlocked,
    machine-written item the active profile@revision did not produce": a
    :class:`~src.services.curation.reprocess_models.ReprocessRequest` (a dry
    run) a client POSTs to ``/reprocess`` verbatim; ``stale_items`` is how
    many items it selects. ``None`` when there is no active profile or
    nothing is stale. Nothing here rewrites an item: the re-run is the
    caller's explicit action.
    """

    items_total: int
    by_profile: list[ActivationImpactByProfile]
    validated_items: int
    unseeded_items: int
    pending_items: int
    pending_not_matching: int
    stale_items: int = 0
    suggested_reprocess: ReprocessRequest | None = None


def _hits_total(resp: dict[str, Any]) -> int:
    raw = resp.get('hits', {}).get('total', 0)
    return int(raw['value']) if isinstance(raw, dict) else int(raw)


def _active_revision(profile: DetectionProfile) -> int | None:
    """The revision the config store has active for ``profile`` (``None``
    for an env/file-registered profile never activated through the store);
    the same stamp the detection worker writes on every region write."""
    from src.services.config_store import get_config_store

    try:
        ref = get_config_store().current.active_profile
    except Exception:
        return None
    return ref[1] if isinstance(ref, tuple) and ref[0] == profile.name else None


async def _suggest_reprocess(
    opensearch: Any, profile: DetectionProfile | None, fields: Any, index: str
) -> tuple[ReprocessRequest | None, int]:
    if profile is None:
        return None, 0
    revision = _active_revision(profile)
    selector = ReprocessFilter(
        profile_not=profile.name, profile_revision_below=revision, include_detected=True
    )
    stale = await opensearch.count(
        index=index,
        body={
            'query': {
                'bool': {
                    'filter': [
                        *selector_clauses(selector, fields),
                        {
                            'terms': {
                                fields.status: [
                                    *(s.value for s in REQUEUEABLE_STATUSES),
                                    RegionStatus.DETECTED.value,
                                ]
                            }
                        },
                    ],
                    'must_not': [{'term': {fields.validated: True}}],
                }
            }
        },
    )
    count = int(stale.get('count', 0))
    if count == 0:
        return None, 0
    return (
        ReprocessRequest(
            targets=ReprocessTargets(filter=selector),
            scopes=['region'],
            region_mode='redetect',
            dry_run=True,
        ),
        count,
    )


async def compute_activation_impact(
    opensearch: Any, *, profile: DetectionProfile | None
) -> ActivationImpact:
    """Aggregate the bound project's items index by ``region_profile``
    (+ sub-terms on ``region_profile_revision``), plus filter counts for
    validated / pending items, and (when ``profile`` is given) pending
    items whose class no longer matches its ``parent_classes``."""
    fields = get_region_fields()
    index = items_index()
    resp = await opensearch.search(
        index=index,
        body={
            'size': 0,
            # M-4 fix (W3/W4 review 2026-09-28): without this, OpenSearch's
            # default 10,000-hit tracking cap silently truncates
            # `items_total` for any project past that size -- see
            # `src/routers/curation/stats.py`'s own note on the same trap
            # ("under-counting at 10k looks like a stuck pipeline").
            'track_total_hits': True,
            'query': {'match_all': {}},
            'aggs': {
                'by_profile': {
                    'terms': {'field': fields.profile, 'size': 200, 'missing': _UNSEEDED_KEY},
                    'aggs': {
                        'by_revision': {
                            'terms': {'field': fields.profile_revision, 'size': 50, 'missing': -1}
                        }
                    },
                },
                'validated': {'filter': {'term': {fields.validated: True}}},
                'pending': {'filter': {'terms': {fields.status: list(_PENDING_STATUSES)}}},
            },
        },
    )
    aggs = resp.get('aggregations', {})
    by_profile: list[ActivationImpactByProfile] = []
    unseeded_items = 0
    for bucket in aggs.get('by_profile', {}).get('buckets', []):
        name = bucket.get('key')
        revision_buckets = bucket.get('by_revision', {}).get('buckets', [])
        if not revision_buckets:
            revision_buckets = [{'key': -1, 'doc_count': bucket.get('doc_count', 0)}]
        for rev_bucket in revision_buckets:
            count = int(rev_bucket.get('doc_count', 0))
            if name == _UNSEEDED_KEY:
                unseeded_items += count
                continue
            revision_key = rev_bucket.get('key')
            revision = None if revision_key in (-1, '-1', None) else int(revision_key)
            by_profile.append(ActivationImpactByProfile(name=name, revision=revision, count=count))

    validated_items = int(aggs.get('validated', {}).get('doc_count', 0))
    pending_items = int(aggs.get('pending', {}).get('doc_count', 0))

    pending_not_matching = 0
    if profile is not None:
        clause = parent_classes_clause(profile.parent_classes)
        if clause is not None:
            count_resp = await opensearch.count(
                index=index,
                body={
                    'query': {
                        'bool': {
                            'filter': [{'terms': {fields.status: list(_PENDING_STATUSES)}}],
                            'must_not': [clause],
                        }
                    }
                },
            )
            pending_not_matching = int(count_resp.get('count', 0))

    suggested, stale_items = await _suggest_reprocess(opensearch, profile, fields, index)
    return ActivationImpact(
        stale_items=stale_items,
        suggested_reprocess=suggested,
        items_total=_hits_total(resp),
        by_profile=by_profile,
        validated_items=validated_items,
        unseeded_items=unseeded_items,
        pending_items=pending_items,
        pending_not_matching=pending_not_matching,
    )


__all__ = ['ActivationImpact', 'ActivationImpactByProfile', 'compute_activation_impact']

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

from src.config import PENDING_STATUSES
from src.config.curation import items_index
from src.config.region_fields import get_region_fields
from src.services.curation.region_eval import LEGACY_STATUS_ALIASES
from src.services.curation.region_scope import parent_classes_clause


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
    """§4.6/§7.3. No ``suggested_reprocess``/``suggested_requeue`` field
    yet (glue G2): that is W10's ``ReprocessRequest`` reference, added at
    merge time by whichever of W4/W10 merges second -- see this wave's
    handback report."""

    items_total: int
    by_profile: list[ActivationImpactByProfile]
    validated_items: int
    unseeded_items: int
    pending_items: int
    pending_not_matching: int


def _hits_total(resp: dict[str, Any]) -> int:
    raw = resp.get('hits', {}).get('total', 0)
    return int(raw['value']) if isinstance(raw, dict) else int(raw)


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

    return ActivationImpact(
        items_total=_hits_total(resp),
        by_profile=by_profile,
        validated_items=validated_items,
        unseeded_items=unseeded_items,
        pending_items=pending_items,
        pending_not_matching=pending_not_matching,
    )


__all__ = ['ActivationImpact', 'ActivationImpactByProfile', 'compute_activation_impact']

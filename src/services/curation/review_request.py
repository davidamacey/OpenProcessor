"""One review-queue request: the query, the sort and where an item sits in it.

``GET /review/{tab}`` and ``GET /review/{tab}/locate`` build the same
request here, so locate always answers for exactly the queue the page
shows. Every sort ends in a ``crop_id`` tiebreak: without one, items with
equal sort values come back in shard order, pages can overlap, and no
count can say where an item is.

Locating counts the items that sort strictly before a target under the
same query — one ``count`` call regardless of queue depth.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from src.services.curation import review_queries, review_sorts
from src.services.curation.item_filter import ItemFilter, item_filter_clauses


TIEBREAK: dict[str, Any] = {'crop_id': {'order': 'asc'}}


@dataclass(frozen=True)
class ReviewFilters:
    include_test: bool = False
    text: str | None = None
    max_rank: int | None = None
    min_blur_ratio: float | None = None
    min_mistakenness: float | None = None
    hide_near_duplicates: bool = False
    source: str | None = None
    # The shared item filter (class names, confidence band, size, origin,
    # embedding state, review status), minus ``max_rank``, which a tab can
    # default and so is read above.
    item: ItemFilter = dataclasses.field(default_factory=ItemFilter)
    region_status: str | None = None
    combine_conflict: bool = False
    import_id: str | None = None
    dataset_split: str | None = None
    on_negative_frame: bool | None = None


@dataclass(frozen=True)
class ReviewRequest:
    query: dict[str, Any]
    sort: list[dict[str, Any]]
    sort_applied: str
    reason: str
    sort_fallback_reason: str | None = None


def _null_safe_floor(field: str, floor: float) -> dict[str, Any]:
    """``field >= floor``, but an item without the field is kept."""
    return {
        'bool': {
            'should': [
                {'range': {field: {'gte': floor}}},
                {'bool': {'must_not': {'exists': {'field': field}}}},
            ],
            'minimum_should_match': 1,
        }
    }


async def build_review_request(
    tab: str, filters: ReviewFilters, sort_id: str | None, opensearch: Any
) -> ReviewRequest:
    """Raises ``HTTPException(400)`` for an unknown tab, ``ValueError`` for
    a bad sort id or confidence band."""
    must, must_not, reason = review_queries.build_tab_query(
        tab,
        include_test=filters.include_test,
        text=filters.text,
        max_rank=filters.max_rank,
        region_status=filters.region_status,
    )
    if filters.min_blur_ratio is not None:
        must.append(_null_safe_floor('blur_lap_ratio', filters.min_blur_ratio))
    if filters.min_mistakenness is not None:
        must.append(_null_safe_floor('mistakenness_score', filters.min_mistakenness))
    if filters.hide_near_duplicates:
        # Hides only items a scoring pass marked as a non-representative
        # duplicate; never-scored items stay.
        must.append(
            {
                'bool': {
                    'should': [
                        {'term': {'dup_is_representative': True}},
                        {'bool': {'must_not': {'exists': {'field': 'dup_is_representative'}}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
    if filters.source:
        must.append({'term': {'source': filters.source}})
    if filters.combine_conflict:
        must.append({'term': {'combine_conflict': True}})
    if filters.on_negative_frame is not None:
        must.append(review_queries.negative_frame_clause(filters.on_negative_frame))
    honoured = review_queries.tab_filters(tab)
    if filters.import_id and 'import_id' in honoured:
        must.append({'term': {'import_ids': filters.import_id}})
    if filters.dataset_split and 'dataset_split' in honoured:
        must.append({'term': {'dataset_split': filters.dataset_split}})
    must.extend(item_filter_clauses(filters.item))
    clause, applied, fallback = await review_sorts.build_sort(
        sort_id, tab=tab, opensearch=opensearch
    )
    return ReviewRequest(
        query={'bool': {'must': must, 'must_not': must_not}},
        sort=[*clause, TIEBREAK],
        sort_applied=applied,
        reason=reason,
        sort_fallback_reason=fallback,
    )


class UnlocatableSortError(ValueError):
    """A sort clause this module can't express as a range query."""


def _sort_key(entry: dict[str, Any] | str) -> tuple[str, str]:
    if isinstance(entry, str):
        return entry, 'asc'
    ((field, spec),) = entry.items()
    if field.startswith('_'):
        raise UnlocatableSortError(f'cannot locate under a {field} sort')
    if isinstance(spec, dict) and spec.get('missing', '_last') != '_last':
        raise UnlocatableSortError(f'cannot locate under missing={spec["missing"]!r}')
    order = spec.get('order', 'asc') if isinstance(spec, dict) else str(spec)
    return field, order


def _equal(field: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {'bool': {'must_not': {'exists': {'field': field}}}}
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {'range': {field: {'gte': value, 'lte': value}}}
    return {'term': {field: value}}


def _strictly_before(field: str, order: str, value: Any) -> dict[str, Any] | None:
    """Items that sort before ``value`` on ``field`` (missing values sort last)."""
    if value is None:
        return {'exists': {'field': field}}
    return {'range': {field: {'lt' if order == 'asc' else 'gt': value}}}


def before_query(sort: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    """Query for items sorting strictly before the doc ``source`` under ``sort``."""
    keys = [_sort_key(e) for e in sort]
    alternatives: list[dict[str, Any]] = []
    for i, (field, order) in enumerate(keys):
        value = source.get(field)
        if isinstance(value, list):
            raise UnlocatableSortError(f'cannot locate by multi-valued field {field}')
        prior = [_equal(f, source.get(f)) for f, _o in keys[:i]]
        alternatives.append({'bool': {'filter': [*prior, _strictly_before(field, order, value)]}})
    return {'bool': {'should': alternatives, 'minimum_should_match': 1}}


__all__ = [
    'TIEBREAK',
    'ReviewFilters',
    'ReviewRequest',
    'UnlocatableSortError',
    'before_query',
    'build_review_request',
]

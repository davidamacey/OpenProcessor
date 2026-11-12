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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.near_dup_fields import near_dup_visible_clause, null_safe_floor
from src.services.curation import review_queries, review_sorts
from src.services.curation.image_join import ImageJoin, resolve_filter_join
from src.services.curation.item_filter import item_filter_clauses
from src.services.embedding_spaces.readers import item_write_fields


if TYPE_CHECKING:
    from src.services.curation.review_filters import ReviewFilters
    from src.services.embedding_spaces.readers import WriteFields


TIEBREAK: dict[str, Any] = {'crop_id': {'order': 'asc'}}


@dataclass(frozen=True)
class ReviewRequest:
    query: dict[str, Any]
    sort: list[dict[str, Any]]
    sort_applied: str
    reason: str
    sort_fallback_reason: str | None = None
    image_join: ImageJoin | None = None


def review_query(
    tab: str, filters: ReviewFilters, embedding: WriteFields, join: ImageJoin | None
) -> tuple[dict[str, Any], str]:
    """``(query, reason)`` of the queue ``tab`` under ``filters``. Raises
    ``RequestRejectedError(400)`` for an unknown tab, ``ValueError`` for a bad confidence band."""
    must, must_not, reason = review_queries.build_tab_query(
        tab,
        include_test=filters.include_test,
        text=filters.text,
        max_rank=filters.max_rank,
        region_status=filters.region_status,
        region_text_match=filters.region_text_match,
        vector_field=embedding.vector_field,
    )
    if filters.min_blur_ratio is not None:
        must.append(null_safe_floor('blur_lap_ratio', filters.min_blur_ratio))
    if filters.min_mistakenness is not None:
        must.append(null_safe_floor('mistakenness_score', filters.min_mistakenness))
    if filters.hide_near_duplicates:
        must.append(near_dup_visible_clause())
    if filters.source:
        must.append({'term': {'source': filters.source}})
    if filters.combine_conflict:
        must.append({'term': {'combine_conflict': True}})
    if filters.on_negative_frame is not None:
        must.append(review_queries.negative_frame_clause(filters.on_negative_frame))
    must.extend(item_filter_clauses(filters.item, embedding=embedding, image_join=join))
    return {'bool': {'must': must, 'must_not': must_not}}, reason


async def build_review_request(
    tab: str, filters: ReviewFilters, sort_id: str | None, opensearch: Any
) -> ReviewRequest:
    """Raises ``RequestRejectedError(400)`` for an unknown tab, ``ValueError`` for
    a bad sort id or confidence band."""
    embedding = await item_write_fields()
    join = await resolve_filter_join(opensearch, filters.item)
    query, reason = review_query(tab, filters, embedding, join)
    clause, applied, fallback = await review_sorts.build_sort(
        sort_id, tab=tab, opensearch=opensearch
    )
    return ReviewRequest(
        query=query,
        sort=[*clause, TIEBREAK],
        sort_applied=applied,
        reason=reason,
        sort_fallback_reason=fallback,
        image_join=join,
    )


__all__ = [
    'TIEBREAK',
    'ReviewRequest',
    'build_review_request',
    'review_query',
]

"""Query building for ``GET /crops``: the filter clauses, the search body and
the ordered-id fetch. Kept out of the router so it stays a thin HTTP layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.curation import items_index
from src.services.curation.crop_browse import parse_crop_sort
from src.services.curation.item_filter import ItemFilter, item_filter_clauses, visibility_clauses
from src.services.curation.wire import item_list_source_excludes, serialize_item


MAX_IDS = 500


@dataclass(frozen=True)
class CropListQuery:
    """The parsed ``GET /crops`` query: sort clause, the merged filter and
    the OpenSearch ``query`` clause built from it."""

    sort_clause: list[dict[str, Any]]
    item_filter: ItemFilter
    query_clause: dict[str, Any]


def build_crop_list_query(
    *,
    sort: str | None,
    item_filter_params: dict[str, Any],
    route_params: dict[str, Any],
    include_test: bool,
    include_excluded: bool,
) -> CropListQuery:
    """Merge the shared filter with the route's own params into one query.

    Raises ``ValueError`` for a bad ``sort`` or filter value.
    """
    sort_clause = parse_crop_sort(sort)
    # The route's own params join the shared filter; one builder reads them all.
    flt = ItemFilter(
        **item_filter_params,
        **{k: v for k, v in route_params.items() if v is not None},
    )
    # Every clause is a pure predicate (term/exists/range/must_not), none
    # scores, so all of it lives in filter context, not must.
    filt = [
        *item_filter_clauses(flt),
        *visibility_clauses(
            include_test=include_test,
            include_excluded=include_excluded or 'excluded' in flt.review_status,
        ),
    ]
    bool_q: dict[str, Any] = {}
    if filt:
        bool_q['filter'] = filt
    query_clause: dict[str, Any] = {'bool': bool_q} if bool_q else {'match_all': {}}
    return CropListQuery(sort_clause=sort_clause, item_filter=flt, query_clause=query_clause)


def crop_list_body(query: CropListQuery, *, page: int, page_size: int) -> dict[str, Any]:
    """The paginated search body for ``query``."""
    return {
        'from': (page - 1) * page_size,
        'size': page_size,
        'query': query.query_clause,
        'sort': query.sort_clause,
        # Exact total (not the default 10k cap) so the labeler shows real
        # queue sizes for filtered views — one count pass per query, fine at
        # this scale and matches the /curation/review endpoint.
        'track_total_hits': True,
        # Never ship the 1024-d embedding vectors or class_id_history
        # to the card grid (history is undo-only).
        '_source': {'excludes': item_list_source_excludes()},
    }


async def crops_by_ids(opensearch: Any, ids: list[str]) -> list[dict[str, Any]]:
    """mget item docs preserving the supplied id order (drops missing)."""
    if not ids:
        return []
    resp = await opensearch.mget(
        index=items_index(),
        body={'ids': ids},
        _source_excludes=item_list_source_excludes(),
    )
    return [
        serialize_item(d.get('_source') or {}, d.get('_id', ''))
        for d in (resp.get('docs') or [])
        if d.get('found')
    ]

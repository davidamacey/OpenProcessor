"""Rows for the region list routes: one row per matched box, or per item.

``GET /regions``, ``GET /regions/training_candidates``, ``GET
/regions/suspected_false_positives``, the region-cluster listings and the
items of ``POST /regions/batch_status`` all return *rows*: the full wire
item (:func:`~src.services.curation.wire.serialize_item`) plus
``region_box_id``, the box the row is about (``None`` for an item-level
row), plus the route's own row keys.

One rule, one function (:func:`region_rows`): when a request selects
**boxes** (a per-box filter, e.g. ``detector=sam3``) each box that matched
is its own row; when it selects only items, one row per item. Which boxes
matched is asked of OpenSearch itself -- the nested clause carries
``inner_hits`` -- so the predicate is never re-implemented in Python and a
multi-filter request always means the *same* box.

``page`` / ``page_size`` page **items** (the unit OpenSearch paginates); a
page returns every row of its items. ``total`` counts items, ``total_rows``
counts rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch import inner_result_window
from src.config import get_region_fields
from src.config.curation import get_curation_config
from src.core.logging import get_logger
from src.services.curation.wire import serialize_item


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch

    from src.config.region_fields import RegionFields

logger = get_logger(__name__)

INNER_HITS_NAME = 'selected_boxes'


# OpenSearch's own default for ``index.max_inner_result_window``; assumed
# until ``ensure_items_inner_result_window`` has read the real value back.
_DEFAULT_INNER_WINDOW = 100


def inner_hits_size(index: str) -> int:
    """How many matched boxes one item reports: the largest box list one
    write may produce, clamped to the index's ``max_inner_result_window``
    (raised to that limit by ``ensure_items_inner_result_window``; the
    OpenSearch default until that has run). A request over the window would
    400, so a clamp -- logged -- is the safe failure."""
    wanted = int(get_curation_config().region_max_boxes_per_write)
    window = inner_result_window(index) or _DEFAULT_INNER_WINDOW
    if window < wanted:
        logger.warning('inner_hits_window_clamped', wanted=wanted, window=window)
        return window
    return wanted


def box_selector(
    clause: dict[str, Any], *, index: str, F: RegionFields | None = None
) -> dict[str, Any]:
    """``clause`` (a per-box predicate) as a nested query that also reports,
    per item, *which* boxes matched. ``_source`` is off: a matched box is
    identified by its offset in the item's own ``region_boxes``."""
    F = F or get_region_fields()
    return {
        'nested': {
            'path': F.boxes,
            'query': clause,
            'inner_hits': {
                'name': INNER_HITS_NAME,
                'size': inner_hits_size(index),
                '_source': False,
            },
        }
    }


def rows_aggs(clause: dict[str, Any], F: RegionFields | None = None) -> dict[str, Any]:
    """The aggregation counting the boxes ``clause`` matches (``total_rows``)."""
    F = F or get_region_fields()
    return {
        'region_rows': {
            'nested': {'path': F.boxes},
            'aggs': {'matching': {'filter': clause}},
        }
    }


def matched_box_ids(hit: dict[str, Any], F: RegionFields | None = None) -> list[str]:
    """The ``box_id`` of every box of ``hit`` that matched the selector, in
    list order."""
    F = F or get_region_fields()
    stored = (hit.get('_source') or {}).get(F.boxes) or []
    inner = ((hit.get('inner_hits') or {}).get(INNER_HITS_NAME) or {}).get('hits', {}).get('hits')
    ids: list[str] = []
    for ih in inner or []:
        offset = (ih.get('_nested') or {}).get('offset')
        if isinstance(offset, int) and 0 <= offset < len(stored):
            box_id = stored[offset].get('box_id')
            if box_id:
                ids.append(box_id)
    return ids


def as_row(item: dict[str, Any], box_id: str | None, **row_keys: Any) -> dict[str, Any]:
    """``item`` (a wire item) as a row about ``box_id`` (``None``: item-level)."""
    return {**item, 'region_box_id': box_id, **row_keys}


def region_rows(
    hits: Sequence[dict[str, Any]],
    *,
    boxes_selected: bool,
    F: RegionFields | None = None,
    row_keys: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The rows of one page of item hits. ``boxes_selected``: one row per
    matched box (an item whose selector matched nothing yields none);
    otherwise one item-level row with ``region_box_id: None``. ``row_keys``
    are merged into every row (a route's ``selection_reason``)."""
    F = F or get_region_fields()
    rows: list[dict[str, Any]] = []
    for hit in hits:
        item = serialize_item(hit.get('_source') or {}, hit.get('_id') or '')
        box_ids: Sequence[str | None] = list(matched_box_ids(hit, F)) if boxes_selected else [None]
        rows.extend(as_row(item, box_id, **(row_keys or {})) for box_id in box_ids)
    return rows


async def rows_for_pairs(
    client: AsyncOpenSearch,
    *,
    index: str,
    pairs: Sequence[tuple[str, str | None, dict[str, Any]]],
    source_excludes: Sequence[str],
) -> list[dict[str, Any]]:
    """Rows for ``(crop_id, region_box_id, row_keys)`` triples, in order, for
    routes that pick their boxes themselves (the FP matcher's scored list, a
    cluster card's representatives). An item that no longer exists yields no
    row."""
    ids = list(dict.fromkeys(crop_id for crop_id, _box, _keys in pairs))
    if not ids:
        return []
    docs = await client.mget(index=index, body={'ids': ids}, _source_excludes=list(source_excludes))
    sources = {d['_id']: d.get('_source') or {} for d in docs['docs'] if d.get('found')}
    return [
        as_row(serialize_item(sources[crop_id], crop_id), box_id, **keys)
        for crop_id, box_id, keys in pairs
        if crop_id in sources
    ]


async def search_region_rows(
    client: AsyncOpenSearch,
    *,
    index: str,
    filters: Sequence[dict[str, Any]],
    must_not: Sequence[dict[str, Any]],
    sort: Sequence[dict[str, Any]],
    page: int,
    page_size: int,
    box_clause: dict[str, Any] | None,
    source_excludes: Sequence[str],
    row_keys: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One page of rows: the ``RegionRowPage`` envelope.

    ``filters`` select items; ``box_clause`` (when given) selects boxes
    *and* items having one, and turns the rows into per-box rows.
    """
    F = get_region_fields()
    query_filters = list(filters)
    body: dict[str, Any] = {
        'from': (page - 1) * page_size,
        'size': page_size,
        '_source': {'excludes': list(source_excludes)},
        'sort': list(sort),
        'track_total_hits': True,
    }
    if box_clause is not None:
        query_filters.append(box_selector(box_clause, index=index, F=F))
        body['aggs'] = rows_aggs(box_clause, F)
    body['query'] = {'bool': {'filter': query_filters, 'must_not': list(must_not)}}
    resp = await client.search(index=index, body=body)
    hits_block = resp.get('hits') or {}
    hits = hits_block.get('hits') or []
    total = int((hits_block.get('total') or {}).get('value', 0))
    rows = region_rows(hits, boxes_selected=box_clause is not None, F=F, row_keys=row_keys)
    if box_clause is None:
        total_rows = total
    else:
        agg = ((resp.get('aggregations') or {}).get('region_rows') or {}).get('matching') or {}
        total_rows = int(agg.get('doc_count', 0))
    return {
        'items': rows,
        'total': total,
        'total_rows': total_rows,
        'page': page,
        'page_size': page_size,
    }


__all__ = [
    'INNER_HITS_NAME',
    'as_row',
    'box_selector',
    'inner_hits_size',
    'matched_box_ids',
    'region_rows',
    'rows_aggs',
    'rows_for_pairs',
    'search_region_rows',
]

"""Curation review-queue router — ``GET /curation/review/*``.

``POST /review/new_class_proposals/resolve`` lives in the sibling
``review_resolve.py`` module, ``GET /review/new_class_proposals/summary``
in ``review_proposals.py`` and the raw-label aggregations in
``review_terms.py``, all registered on this same shared ``router``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import HTTPException, Path as PathParam, Query

from src.config import get_region_fields
from src.routers.curation._common import (
    OpenSearchDep,
    get_class_registry,
    guard_page_depth,
    is_not_found,
    items_index,
    router,
)
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.routers.curation._page_models import ReviewLocateResponse, ReviewQueuePage
from src.routers.curation._review_params import (  # noqa: TC001 - FastAPI resolves the aliases
    BlurQ,
    CombineConflictQ,
    DatasetSplitQ,
    ImportIdQ,
    IncludeTest,
    MistakeQ,
    NearDupQ,
    OnNegativeFrameQ,
    RegionStatusQ,
    SortQ,
    SourceQ,
    TextQ,
)
from src.routers.curation._review_tab_models import ReviewTabsResponse
from src.services.curation import review_empty_reason, review_queries
from src.services.curation.index_bootstrap import ensure_indexes
from src.services.curation.review_filter_specs import REGION_STATUS_DEFAULT
from src.services.curation.review_request import (
    ReviewFilters,
    UnlocatableSortError,
    before_query,
    build_review_request,
)
from src.services.curation.wire import item_list_source_excludes, serialize_item


if TYPE_CHECKING:
    from src.services.curation.item_filter import ItemFilter


_TAB_DESCRIPTION = 'One of: ' + ' | '.join(review_queries.KNOWN_TABS)


def _filters(
    include_test: bool,
    text: str | None,
    item_filter: ItemFilter,
    min_blur_ratio: float | None,
    min_mistakenness: float | None,
    hide_near_duplicates: bool,
    source: str | None,
    region_status: str | None,
    combine_conflict: bool,
    import_id: str | None,
    dataset_split: str | None,
    on_negative_frame: bool | None,
) -> ReviewFilters:
    return ReviewFilters(
        include_test=include_test,
        text=text,
        max_rank=item_filter.max_rank,
        min_blur_ratio=min_blur_ratio,
        min_mistakenness=min_mistakenness,
        hide_near_duplicates=hide_near_duplicates,
        source=source,
        item=item_filter.model_copy(update={'max_rank': None}),
        region_status=region_status,
        combine_conflict=combine_conflict,
        import_id=import_id,
        dataset_split=dataset_split,
        on_negative_frame=on_negative_frame,
    )


async def _request(tab: str, filters: ReviewFilters, sort: str | None, opensearch: Any) -> Any:
    try:
        return await build_review_request(tab, filters, sort, opensearch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/review/tabs', response_model=ReviewTabsResponse)
async def review_tabs(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Every review tab's ``id``/``label``/``description`` plus
    ``filters`` (the query parameters it honours)
    ``filter_defaults`` (values it applies when one is omitted) and
    ``filter_specs`` (self-describing enum filters) — the frontend renders this instead of hardcoding tab labels or assuming a
    filter works everywhere. Must be registered before ``GET /review/{tab}``
    so it isn't shadowed as ``tab='tabs'``.

    Also carries ``empty_state`` -- ``{has_probe_predictions,
    has_item_scores}`` -- computed once here so a client can annotate any
    tab's zero-result state (the same reasons ``GET /review/{tab}``'s own
    ``empty_reason`` uses) without a per-tab round trip.
    """
    empty_state = await review_empty_reason.review_tabs_empty_state(opensearch)
    return {'tabs': review_queries.review_tab_catalog(), 'empty_state': empty_state}


@router.get('/review/{tab}', response_model=None, responses={200: {'model': ReviewQueuePage}})
async def review_queue(
    tab: Annotated[str, PathParam(description=_TAB_DESCRIPTION)],
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 30,
    include_test: IncludeTest = False,
    text: TextQ = None,
    min_blur_ratio: BlurQ = None,
    min_mistakenness: MistakeQ = None,
    hide_near_duplicates: NearDupQ = False,
    source: SourceQ = None,
    sort: SortQ = None,
    region_status: RegionStatusQ = REGION_STATUS_DEFAULT,
    combine_conflict: CombineConflictQ = False,
    import_id: ImportIdQ = None,
    dataset_split: DatasetSplitQ = None,
    on_negative_frame: OnNegativeFrameQ = None,
) -> dict[str, Any]:
    """Human review queue for the labeler ``/review`` page.

    Each tab maps to a deterministic OpenSearch query against the
    configured items index plus a ``reason`` string the labeler renders to
    explain why the crop landed in this queue. Items already validated by
    a human are excluded from every tab — except ``model_disagreements``,
    where validated crops are exactly the input set, and ``regions``,
    which reviews the region annotation independently of the item's
    class validation. Order: the applied sort (``sort_applied``), then
    ``crop_id``.
    """
    await ensure_indexes(opensearch)
    guard_page_depth(page, page_size)
    if review_empty_reason.region_queue_is_off(tab):
        return {
            'total': 0,
            'page': page,
            'page_size': page_size,
            'items': [],
            'sort_applied': sort or 'default',
            'sort_fallback_reason': None,
            'empty_reason': review_empty_reason.REGION_PROFILE_OFF_REASON,
        }
    filters = _filters(
        include_test,
        text,
        item_filter,
        min_blur_ratio,
        min_mistakenness,
        hide_near_duplicates,
        source,
        region_status,
        combine_conflict,
        import_id,
        dataset_split,
        on_negative_frame,
    )
    req = await _request(tab, filters, sort, opensearch)
    body = {
        'from': (page - 1) * page_size,
        'size': page_size,
        'query': req.query,
        'sort': req.sort,
        # Exact totals: the default 10k cap makes large queues look smaller.
        'track_total_hits': True,
        # Never ship the 1024-d embedding vectors or class_id_history to
        # the review grid (history is undo-only).
        '_source': {'excludes': item_list_source_excludes()},
    }
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc

    total_obj = (resp.get('hits') or {}).get('total') or {}
    total = int(total_obj.get('value', 0))
    hits = (resp.get('hits') or {}).get('hits') or []
    registry_names: frozenset[str] = frozenset()
    if tab == 'mismatches':
        from src.services.curation.new_class_terms import normalize_term

        registry_names = frozenset(
            normalize_term(c.class_name)
            for c in get_class_registry().load().classes
            if not c.deprecated
        )
    items: list[dict[str, Any]] = []
    region_fields = get_region_fields() if tab == 'regions' else None
    for h in hits:
        src = h.get('_source') or {}
        item = serialize_item(src, h.get('_id', ''))
        # Review-only extra on top of the shared wire item.
        if tab == 'mismatches':
            item['reason'] = review_queries.mismatch_reason(src, registry_names, req.reason)
        elif tab == 'regions':
            item['reason'] = review_queries.region_reason(src, region_fields, req.reason)
        else:
            item['reason'] = req.reason
        items.append(item)
    # A real-state reason a bare "Queue empty." can't tell the
    # operator -- only computed on the empty path (a few extra `count`
    # calls that only fire when there's nothing else to show anyway).
    empty_reason = (
        await review_empty_reason.compute_empty_reason(tab, filters, opensearch)
        if total == 0
        else None
    )
    return {
        'total': int(total),
        'page': page,
        'page_size': page_size,
        'items': items,
        'sort_applied': req.sort_applied,
        # Set when the default sort's field has 0% coverage and
        # sort_applied is its fallback.
        'sort_fallback_reason': req.sort_fallback_reason,
        'empty_reason': empty_reason,
    }


@router.get(
    '/review/{tab}/locate', response_model=None, responses={200: {'model': ReviewLocateResponse}}
)
async def review_locate(
    tab: Annotated[str, PathParam(description=_TAB_DESCRIPTION)],
    crop_id: Annotated[str, Query(description='The item to find.')],
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    page_size: Annotated[int, Query(ge=1, le=200)] = 30,
    include_test: IncludeTest = False,
    text: TextQ = None,
    min_blur_ratio: BlurQ = None,
    min_mistakenness: MistakeQ = None,
    hide_near_duplicates: NearDupQ = False,
    source: SourceQ = None,
    sort: SortQ = None,
    region_status: RegionStatusQ = REGION_STATUS_DEFAULT,
    combine_conflict: CombineConflictQ = False,
    import_id: ImportIdQ = None,
    dataset_split: DatasetSplitQ = None,
    on_negative_frame: OnNegativeFrameQ = None,
) -> dict[str, Any]:
    """Where ``crop_id`` sits in the queue ``GET /review/{tab}`` would serve
    for the same filters and sort.

    ``{crop_id, in_queue, rank, page, page_size, total, reason,
    sort_applied, sort_fallback_reason}``: ``rank`` is 0-based, ``page`` the 1-based page of
    ``page_size`` holding it. Out of the queue: ``rank``/``page`` null and
    ``reason`` ``not_found`` (no such item) or ``filtered_out``. Counts the
    items sorting before it, so any depth costs the same.
    """
    await ensure_indexes(opensearch)
    filters = _filters(
        include_test,
        text,
        item_filter,
        min_blur_ratio,
        min_mistakenness,
        hide_near_duplicates,
        source,
        region_status,
        combine_conflict,
        import_id,
        dataset_split,
        on_negative_frame,
    )
    req = await _request(tab, filters, sort, opensearch)
    if review_empty_reason.region_queue_is_off(tab):
        return {
            'crop_id': crop_id,
            'in_queue': False,
            'rank': None,
            'page': None,
            'page_size': page_size,
            'total': 0,
            'reason': 'region_profile_off',
            'sort_applied': req.sort_applied,
            'sort_fallback_reason': req.sort_fallback_reason,
        }
    out: dict[str, Any] = {
        'crop_id': crop_id,
        'in_queue': False,
        'rank': None,
        'page': None,
        'page_size': page_size,
        'total': None,
        'reason': None,
        'sort_applied': req.sort_applied,
        'sort_fallback_reason': req.sort_fallback_reason,
    }

    async def _count(query: dict[str, Any]) -> int:
        try:
            resp = await opensearch.count(index=items_index(), body={'query': query})
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
        return int(resp.get('count', 0))

    try:
        doc = await opensearch.get(index=items_index(), id=crop_id)
    except Exception as exc:
        if is_not_found(exc):
            return {**out, 'reason': 'not_found'}
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    if not doc.get('found', True):
        return {**out, 'reason': 'not_found'}
    source_doc = {**(doc.get('_source') or {}), 'crop_id': crop_id}
    out['total'] = await _count(req.query)
    member = {'bool': {'filter': [req.query, {'term': {'crop_id': crop_id}}]}}
    if not await _count(member):
        return {**out, 'reason': 'filtered_out'}
    try:
        before = before_query(req.sort, source_doc)
    except UnlocatableSortError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rank = await _count({'bool': {'filter': [req.query, before]}})
    return {**out, 'in_queue': True, 'rank': rank, 'page': rank // page_size + 1}

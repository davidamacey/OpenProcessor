"""Curation review-queue router — ``GET /curation/review/*``.

``POST /review/new_class_proposals/resolve`` lives in the sibling
``review_resolve.py`` module, ``GET /review/new_class_proposals/summary``
in ``review_proposals.py`` and the raw-label aggregations in
``review_terms.py``, all registered on this same shared ``router``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Path as PathParam, Query

from src.config import get_region_fields
from src.config.curation_accessors import items_index
from src.routers._errors import server_error
from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, get_class_registry, is_not_found, router
from src.routers.curation._cursor_params import CursorQ, cursor_refused
from src.routers.curation._item_filter_params import ItemFilterQuery
from src.routers.curation._page_models import ReviewLocateResponse, ReviewQueuePage
from src.routers.curation._review_params import (
    BlurQ,
    CombineConflictQ,
    IncludeTest,
    MistakeQ,
    NearDupQ,
    OnNegativeFrameQ,
    RegionStatusQ,
    RegionTextMatchQ,
    SortQ,
    SourceQ,
    TextQ,
)
from src.routers.curation._review_tab_models import ReviewTabsResponse
from src.schemas.curation.config_common import ApiErrorResponse
from src.services.curation import review_empty_reason, review_reasons, review_sorts
from src.services.curation.audit import has_pending_sample
from src.services.curation.hidden_counts import hidden_counts
from src.services.curation.image_join import ImageJoin, join_truncated
from src.services.curation.item_filter import item_filter_clauses
from src.services.curation.list_cursor import CursorError, page_body, split_page
from src.services.curation.list_locate import UnlocatableSortError, locate_item
from src.services.curation.review_filters import REGION_STATUS_DEFAULT, ReviewFilters
from src.services.curation.review_request import build_review_request
from src.services.curation.review_tab_status import absent_queues, tab_counts
from src.services.curation.review_tabs import QUEUES, review_tab_catalog
from src.services.curation.wire import item_list_source_excludes, serialize_item
from src.services.embedding_spaces.readers import (
    item_state_fields,
    item_write_fields,
    vector_source_excludes,
)


_TAB_DESCRIPTION = 'One of: ' + ' | '.join(QUEUES)
_NOT_A_QUEUE: dict[int | str, dict[str, Any]] = {404: {'model': ApiErrorResponse}}


def _review_filters(
    item_filter: ItemFilterQuery,
    include_test: IncludeTest = False,
    text: TextQ = None,
    region_text_match: RegionTextMatchQ = None,
    min_blur_ratio: BlurQ = None,
    min_mistakenness: MistakeQ = None,
    hide_near_duplicates: NearDupQ = False,
    source: SourceQ = None,
    region_status: RegionStatusQ = REGION_STATUS_DEFAULT,
    combine_conflict: CombineConflictQ = False,
    on_negative_frame: OnNegativeFrameQ = None,
) -> ReviewFilters:
    return ReviewFilters(
        include_test=include_test,
        text=text,
        region_text_match=region_text_match,
        max_rank=item_filter.max_rank,
        min_blur_ratio=min_blur_ratio,
        min_mistakenness=min_mistakenness,
        hide_near_duplicates=hide_near_duplicates,
        source=source,
        item=item_filter.model_copy(update={'max_rank': None}),
        region_status=region_status,
        combine_conflict=combine_conflict,
        on_negative_frame=on_negative_frame,
    )


ReviewFiltersDep = Annotated[ReviewFilters, Depends(_review_filters)]


def _queue(
    tab: Annotated[str, PathParam(description=_TAB_DESCRIPTION)],
) -> str:
    """The queue a ``/review/{tab}`` path names. The other tab ids are ``review_tab`` filter values
    only, so they answer 404 here."""
    if tab not in QUEUES:
        raise api_error(
            404,
            'not_a_queue',
            f'{tab!r} is not a review queue; choose from {", ".join(QUEUES)}. '
            'Filter any list with review_tab instead.',
        )
    return tab


QueueDep = Annotated[str, Depends(_queue)]


async def _request(tab: str, filters: ReviewFilters, sort: str | None, opensearch: Any) -> Any:
    try:
        return await build_review_request(tab, filters, sort, opensearch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _hidden(
    opensearch: Any, filters: ReviewFilters, join: ImageJoin | None
) -> dict[str, int]:
    """Review queues never show ignored items; the test holdout only with ``include_test``."""
    try:
        clauses = item_filter_clauses(
            filters.item, embedding=await item_write_fields(), image_join=join
        )
        return await hidden_counts(
            opensearch, clauses, include_test=filters.include_test, include_excluded=False
        )
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc


@router.get('/review/tabs', response_model=ReviewTabsResponse)
async def review_tabs(opensearch: OpenSearchDep, filters: ReviewFiltersDep) -> dict[str, Any]:
    """The review queues in order: ``id``, ``label``, ``description``, the ``filters`` each honours,
    their ``filter_defaults`` and self-describing ``filter_specs``, and ``count``, the items the
    queue holds under the filter the request carries (the same filter ``GET /review/{queue}``
    takes). A queue that cannot fill is not listed: the audit queue exists only while a drawn
    sample is pending, the region queue only while a region profile is active, and
    ``GET /next_steps`` says what to do. Must be registered before ``GET /review/{tab}`` so it
    isn't shadowed as ``tab='tabs'``.

    Also carries ``empty_state`` -- ``{has_probe_predictions, has_item_scores,
    has_unembedded_items, suggested_reprocess}`` -- computed once here so a client can annotate any
    queue's zero-result state (the same reasons ``GET /review/{queue}``'s own
    ``empty_reason`` uses) without a per-queue round trip.
    """
    empty_state = await review_empty_reason.review_tabs_empty_state(opensearch)
    try:
        absent = absent_queues(has_pending_audit=await has_pending_sample(opensearch))
        queues = [t for t in review_tab_catalog() if t['id'] not in absent]
        counts = await tab_counts(opensearch, [t['id'] for t in queues], filters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc
    return {
        'tabs': [{**t, 'count': counts[t['id']]} for t in queues],
        'empty_state': empty_state,
    }


@router.get(
    '/review/{tab}',
    response_model=None,
    responses={200: {'model': ReviewQueuePage}, **_NOT_A_QUEUE},
)
async def review_queue(
    tab: QueueDep,
    opensearch: OpenSearchDep,
    filters: ReviewFiltersDep,
    cursor: CursorQ = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 30,
    sort: SortQ = None,
) -> dict[str, Any]:
    """One review queue: a deterministic OpenSearch query against the items index plus a ``reason``
    string that explains why the crop landed in it. Items a human already validated are excluded
    from every queue, except the model-disagreement source of ``disagreements`` (validated crops
    are exactly its input) and ``regions``, which reviews the region annotation independently of
    the item's class validation. Order: the applied sort (``sort_applied``), then ``crop_id``. A
    tab id that is not a queue answers 404 ``not_a_queue``.
    """
    if review_empty_reason.region_queue_is_off(tab):
        return {
            'total': 0,
            'next_cursor': None,
            'limit': limit,
            'items': [],
            'sort_applied': sort or 'default',
            'sort_options': [],
            'sort_fallback_reason': None,
            'empty_reason': review_empty_reason.REGION_PROFILE_OFF_REASON,
            # The off profile serves no queue, so nothing is hidden from it.
            'hidden': {'test_holdout': 0, 'excluded': 0},
            'join_truncated': False,
        }
    req = await _request(tab, filters, sort, opensearch)
    sort_options = await review_sorts.sort_options(tab, opensearch)
    excludes = await vector_source_excludes()
    try:
        body = page_body(
            {
                'query': req.query,
                # Exact totals: the default 10k cap makes large queues look smaller.
                'track_total_hits': True,
                # Never ship the 1024-d embedding vectors or class_id_history to
                # the review grid (history is undo-only).
                '_source': {'excludes': item_list_source_excludes(excludes)},
            },
            sort=req.sort,
            tiebreak='crop_id',
            cursor=cursor,
            sort_id=req.sort_applied,
            limit=limit,
        )
    except CursorError as exc:
        raise cursor_refused(exc) from exc
    hidden = await _hidden(opensearch, filters, req.image_join)
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc

    total_obj = (resp.get('hits') or {}).get('total') or {}
    total = int(total_obj.get('value', 0))
    hits, next_cursor = split_page(
        (resp.get('hits') or {}).get('hits') or [], sort_id=req.sort_applied, limit=limit
    )
    registry_names: frozenset[str] = frozenset()
    if tab == 'disagreements':
        from src.services.curation.new_class_terms import normalize_term

        registry_names = frozenset(
            normalize_term(c.class_name)
            for c in (await get_class_registry().snapshot()).classes
            if not c.deprecated
        )
    items: list[dict[str, Any]] = []
    region_fields = get_region_fields() if tab == 'regions' else None
    state_fields = await item_state_fields()
    for h in hits:
        src = h.get('_source') or {}
        item = serialize_item(src, h.get('_id', ''), state_fields=state_fields)
        # Review-only extra on top of the shared wire item.
        if tab == 'disagreements':
            item['reason'] = review_reasons.disagreement_reason(src, registry_names)
        elif tab == 'regions':
            item['reason'] = review_reasons.region_reason(src, region_fields, req.reason)
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
        'next_cursor': next_cursor,
        'limit': limit,
        'items': items,
        'sort_applied': req.sort_applied,
        'sort_options': sort_options,
        # Set when the default sort's field has 0% coverage and
        # sort_applied is its fallback.
        'sort_fallback_reason': req.sort_fallback_reason,
        'empty_reason': empty_reason,
        'hidden': hidden,
        'join_truncated': join_truncated(req.image_join),
    }


@router.get(
    '/review/{tab}/locate',
    response_model=None,
    responses={200: {'model': ReviewLocateResponse}, **_NOT_A_QUEUE},
)
async def review_locate(
    tab: QueueDep,
    crop_id: Annotated[str, Query(description='The item to find.')],
    opensearch: OpenSearchDep,
    filters: ReviewFiltersDep,
    sort: SortQ = None,
) -> dict[str, Any]:
    """Where ``crop_id`` sits in the queue ``GET /review/{tab}`` would serve
    for the same filters and sort.

    ``{crop_id, in_queue, index, total, cursor_before, cursor_after, reason,
    sort_applied, sort_fallback_reason}``: ``index`` is the 0-based position. ``cursor_before``
    starts a page at the item (null for the first row), ``cursor_after`` resumes right after
    it. Out of the queue: ``index`` and the cursors null and ``reason`` ``not_found`` (no such
    item) or ``filtered_out``. Counts the items sorting before it, so any depth costs the same.
    """
    req = await _request(tab, filters, sort, opensearch)
    out: dict[str, Any] = {
        'crop_id': crop_id,
        'in_queue': False,
        'index': None,
        'total': 0,
        'cursor_before': None,
        'cursor_after': None,
        'reason': None,
        'sort_applied': req.sort_applied,
        'sort_fallback_reason': req.sort_fallback_reason,
    }
    if review_empty_reason.region_queue_is_off(tab):
        return {**out, 'reason': 'region_profile_off'}
    try:
        doc = await opensearch.get(index=items_index(), id=crop_id)
    except Exception as exc:
        if is_not_found(exc):
            return {**out, 'reason': 'not_found'}
        raise server_error(503, 'opensearch unavailable', exc) from exc
    if not doc.get('found', True):
        return {**out, 'reason': 'not_found'}
    source_doc = {**(doc.get('_source') or {}), 'crop_id': crop_id}
    try:
        where = await locate_item(
            opensearch,
            index=items_index(),
            query=req.query,
            sort=req.sort,
            sort_id=req.sort_applied,
            source=source_doc,
        )
    except UnlocatableSortError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc
    if where is None:
        total = int(
            (await opensearch.count(index=items_index(), body={'query': req.query}))['count']
        )
        return {**out, 'total': total, 'reason': 'filtered_out'}
    return {**out, 'in_queue': True, **where.as_wire()}

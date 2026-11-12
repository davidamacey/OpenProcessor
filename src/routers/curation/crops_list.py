"""Curation router sub-module — item (crop) browse: ``GET /crops`` and ``GET /crops/{crop_id}``."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.config.curation_accessors import items_index
from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, is_not_found, router
from src.routers.curation._crop_params import CropBrowseQuery, resolve_listing
from src.routers.curation._cursor_params import CursorQ, cursor_refused
from src.routers.curation._item_filter_params import ItemFilterQuery
from src.routers.curation._item_models import CropLocate, CropsPageResponse, ItemDoc
from src.services.curation.capture_wire import image_capture
from src.services.curation.cluster_cards import attach_item_display_nos
from src.services.curation.crop_browse import crops_page
from src.services.curation.crop_listing import MAX_IDS, crop_list_body, crops_by_ids
from src.services.curation.crop_orders import ordered_crops_page
from src.services.curation.hidden_counts import hidden_counts
from src.services.curation.image_join import join_truncated
from src.services.curation.list_cursor import CursorError, split_page, with_tiebreak
from src.services.curation.list_locate import locate_item
from src.services.curation.wire import item_source_excludes, serialize_item
from src.services.embedding_spaces.readers import item_state_fields, vector_source_excludes


@router.get('/crops', response_model=None, responses={200: {'model': CropsPageResponse}})
async def list_crops(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    browse: CropBrowseQuery,
    # Annotated defaults (not `= Query(...)`) so direct Python callers such
    # as GET /classes/{id}/crops get real values, not FieldInfo objects.
    cursor: CursorQ = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ids: Annotated[
        str | None,
        Query(
            description=(
                'Comma-separated crop ids (max 500): return exactly these items in this '
                'order, missing ids dropped. Every other filter is ignored.'
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Cursor-paged crop browse with the standard filter set.

    ``next_cursor`` resumes the same sort; the page is as deep as the list. The computed orders
    ``order=outliers`` and ``order=diverse`` rank the matched pool in memory and serve one bounded
    page of at most ``limit`` items with ``next_cursor: null``; ``order=core_first`` pages by cursor.

    ``test_holdout=true`` rows are filtered out unless ``include_test``.
    ``class_excluded=true`` (human-ignored) rows are filtered out unless
    ``include_excluded`` — set it to review the ignore bucket.

    Primary-subject filters (all default off → no behavior change):
    ``max_rank`` keeps only crops whose ``crop_rank_in_image <= max_rank``
    (e.g. 1 = largest only, 2 = largest + 2nd). ``min_blur_ratio`` keeps crops
    at or above a clarity threshold (the labeler slider); crops with no blur
    score are NOT dropped. ``classifier_conf_lt`` mines the "model wasn't sure" pool —
    crops a classifier scored below the value (``confidence``) OR that have no classifier
    prediction at all (blind spots).
    """
    if ids is not None:
        wanted = list(dict.fromkeys(i for i in (x.strip() for x in ids.split(',')) if i))
        if len(wanted) > MAX_IDS:
            raise HTTPException(status_code=400, detail=f'at most {MAX_IDS} ids per request')
        found = await crops_by_ids(opensearch, wanted)
        return crops_page(
            total=len(found), next_cursor=None, limit=len(wanted), crops=found, sort_applied='ids'
        )
    listing = await resolve_listing(opensearch, item_filter, browse)
    narrowed = not listing.item_filter.is_empty()
    try:
        hidden = await hidden_counts(
            opensearch,
            listing.filter_clauses,
            include_test=browse.include_test,
            include_excluded=browse.include_excluded
            or 'excluded' in listing.item_filter.review_status,
        )
        if browse.order != 'default':
            # Computed orders (outliers / core_first / diverse) rank the matched pool; None
            # falls through to the plain sort below.
            n_pool = int(
                (await opensearch.count(index=items_index(), body={'query': listing.query_clause}))[
                    'count'
                ]
            )
            ordered = await ordered_crops_page(
                opensearch,
                index=items_index(),
                order=browse.order,
                query_clause=listing.query_clause,
                cluster_uid=str(item_filter.cluster_uids[0])
                if len(item_filter.cluster_uids) == 1
                else None,
                item_filter=listing.item_filter,
                cursor=cursor,
                limit=limit,
                k=browse.k,
                n_pool=n_pool,
                fetch_items=crops_by_ids,
            )
            if ordered is not None:
                await attach_item_display_nos(ordered['crops'])
                ordered['hidden'] = hidden
                ordered['join_truncated'] = join_truncated(listing.image_join)
                return ordered
        body = crop_list_body(
            listing, cursor=cursor, limit=limit, vector_excludes=await vector_source_excludes()
        )
        resp = await opensearch.search(index=items_index(), body=body)
    except CursorError as exc:
        raise cursor_refused(exc) from exc
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc
    total = int((resp.get('hits') or {}).get('total', {}).get('value', 0))
    hits, next_cursor = split_page(
        (resp.get('hits') or {}).get('hits') or [], sort_id=listing.sort_id, limit=limit
    )
    state_fields = await item_state_fields()
    crops = [
        serialize_item(h.get('_source') or {}, h.get('_id', ''), state_fields=state_fields)
        for h in hits
    ]
    await attach_item_display_nos(crops)
    return crops_page(
        total=total,
        next_cursor=next_cursor,
        limit=limit,
        crops=crops,
        sort_applied=listing.sort_id,
        narrowed=narrowed,
        hidden=hidden,
        join_truncated=join_truncated(listing.image_join),
    )


@router.get('/crops/locate', response_model=None, responses={200: {'model': CropLocate}})
async def locate_crop(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    browse: CropBrowseQuery,
    crop_id: Annotated[str, Query(description='The crop to find.')],
) -> dict[str, Any]:
    """Where ``crop_id`` sits in the list ``GET /crops`` serves for the same filters and sort.

    ``index`` is the 0-based position; ``cursor_before`` starts a page at the crop (null for the
    first row) and ``cursor_after`` resumes right after it, so a deep link loads one page whatever
    the depth. Not in the list: ``index`` and the cursors are null and ``reason`` is
    ``not_found`` (no such crop) or ``filtered_out``. The computed orders have no stable position
    and answer 400.
    """
    if browse.order != 'default':
        raise HTTPException(
            status_code=400, detail=f'cannot locate a crop under order={browse.order}'
        )
    listing = await resolve_listing(opensearch, item_filter, browse)
    out: dict[str, Any] = {
        'crop_id': crop_id,
        'in_view': False,
        'index': None,
        'total': 0,
        'cursor_before': None,
        'cursor_after': None,
        'reason': None,
        'reason_message': None,
    }
    try:
        out['total'] = int(
            (await opensearch.count(index=items_index(), body={'query': listing.query_clause}))[
                'count'
            ]
        )
        try:
            doc = await opensearch.get(index=items_index(), id=crop_id)
        except Exception as exc:
            if not is_not_found(exc):
                raise
            doc = {'found': False}
        if not doc.get('found', True):
            reason, message = 'not_found', 'No such crop.'
        else:
            where = await locate_item(
                opensearch,
                index=items_index(),
                query=listing.query_clause,
                sort=with_tiebreak(listing.sort_clause, 'crop_id'),
                sort_id=listing.sort_id,
                source={**(doc.get('_source') or {}), 'crop_id': crop_id},
            )
            if where is not None:
                return {**out, 'in_view': True, **where.as_wire()}
            reason, message = 'filtered_out', 'The crop is not in the current filtered view.'
    except HTTPException:
        raise
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc
    return {**out, 'reason': reason, 'reason_message': message}


@router.get('/crops/{crop_id}', response_model=None, responses={200: {'model': ItemDoc}})
async def get_crop(
    crop_id: str,
    opensearch: OpenSearchDep,
) -> dict[str, Any]:
    """Return the authoritative item document by id.

    Used by the labeler's review-queue "Back" path so the operator sees
    what was actually persisted (not a stale local snapshot). Same wire
    item as ``GET /crops`` and ``GET /review/{tab}`` — never the raw
    OpenSearch ``_source``, whose region keys follow ``RegionFields``
    storage names.
    """
    excludes = item_source_excludes(await vector_source_excludes())
    try:
        # Still excludes the 1024-d embedding vectors (matching
        # every other item endpoint's behavior) -- but not
        # class_id_history, since a single-item view may legitimately
        # want it, unlike a paginated list.
        resp = await opensearch.get(index=items_index(), id=crop_id, _source_excludes=excludes)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    src = (resp.get('_source') or {}) if isinstance(resp, dict) else {}
    item = serialize_item(src, crop_id, state_fields=await item_state_fields())
    if (capture := await image_capture(opensearch, str(src.get('image_id') or ''))) is not None:
        item['capture'] = capture
    return item

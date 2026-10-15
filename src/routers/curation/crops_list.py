"""Curation router sub-module — item (crop) browse: ``GET /crops`` and ``GET /crops/{crop_id}``."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.routers.curation._common import (
    CropsPageResponse,
    ItemDoc,
    OpenSearchDep,
    guard_page_depth,
    items_index,
    router,
)
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.routers.curation._review_params import UnsetIfEmpty
from src.services.curation.crop_browse import crops_page
from src.services.curation.crop_listing import (
    MAX_IDS,
    build_crop_list_query,
    crop_list_body,
    crops_by_ids,
)
from src.services.curation.crop_orders import ordered_crops_page
from src.services.curation.diverse_limits import DIVERSE_BROWSE_MAX_K
from src.services.curation.index_bootstrap import ensure_indexes
from src.services.curation.wire import item_source_excludes, serialize_item


@router.get('/crops', response_model=None, responses={200: {'model': CropsPageResponse}})
async def list_crops(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    # Annotated defaults (not `= Query(...)`) so direct Python callers such
    # as GET /classes/{id}/crops get real values, not FieldInfo objects.
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 50,
    limit: Annotated[
        int | None, Query(ge=1, le=500, description='Alias for page_size; wins when both set.')
    ] = None,
    sort: Annotated[
        str | None,
        Query(
            description=(
                "'<field>[:asc|desc]', default 'updated_at:desc'. Fields: "
                'updated_at, created_at, confidence, '
                'crop_rank_in_image, crop_area_norm, blur_lap_ratio, cluster_distance, '
                'mistakenness_score, uniqueness_score. Ignored by order=outliers|diverse.'
            )
        ),
    ] = None,
    cluster_id: int | None = None,
    cluster_subid: Annotated[
        str | None,
        UnsetIfEmpty,
        Query(description="One AHC sub-cluster of a cluster, e.g. '10000b'."),
    ] = None,
    label_source: str | None = None,
    class_source: str | None = None,
    label_validated: bool | None = None,
    import_id: Annotated[
        str | None,
        Query(description='Items a dataset import labeled (`import_ids`) or proposed.'),
    ] = None,
    dataset_split: Annotated[
        str | None,
        UnsetIfEmpty,
        Query(description='The split a dataset import filed the frame under.'),
    ] = None,
    on_negative_frame: Annotated[
        bool | None,
        UnsetIfEmpty,
        Query(description='true = machine items on an imported reviewed-negative frame.'),
    ] = None,
    proposed_by_import: Annotated[
        bool | None,
        Query(description='true = items an import created from the detector; false = the rest.'),
    ] = None,
    source: Annotated[str | None, Query(description='Ingest source tag (wire `source`).')] = None,
    open_vocab_set: Annotated[
        str | None, Query(description='Items the full-image pass wrote for this prompt set.')
    ] = None,
    source_prompt: Annotated[
        str | None, Query(description='Items the full-image pass found with this exact prompt.')
    ] = None,
    needs_new_class: bool | None = None,
    review_dismissed: Annotated[
        bool | None, Query(description='true = only items hidden from review.')
    ] = None,
    ids: Annotated[
        str | None,
        Query(
            description=(
                'Comma-separated crop ids (max 500): return exactly these items in this '
                'order, missing ids dropped. Every other filter is ignored.'
            )
        ),
    ] = None,
    include_test: bool = False,
    include_excluded: bool = False,
    min_blur_ratio: Annotated[float | None, Query(ge=0.0)] = None,
    classifier_conf_lt: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    item_text: Annotated[
        str | None,
        Query(
            max_length=200,
            description=(
                'Text read on the item crop: every word must be a case-insensitive '
                'prefix of a stored item text token.'
            ),
        ),
    ] = None,
    order: Annotated[
        str,
        Query(
            description=(
                "'outliers' ranks a cluster's members farthest-from-centroid first; "
                "'core_first' nearest-first, with each item's cluster_distance / "
                'cluster_similarity / cluster_is_core recomputed against the same live '
                'centroid (both need cluster_id). '
                "'diverse' ranks the matched pool by k-center-greedy coverage "
                '(gated on OP_SELECT_DIVERSE_ENABLED; behaves like an unrecognized '
                'order value when the flag is off).'
            )
        ),
    ] = 'default',
    k: Annotated[
        int | None,
        Query(
            ge=1,
            le=DIVERSE_BROWSE_MAX_K,
            description='order=diverse only: rank just the first k picks.',
        ),
    ] = None,
) -> dict[str, Any]:
    """Paginated crop browse with the standard filter set.

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
    await ensure_indexes(opensearch)
    if ids is not None:
        wanted = list(dict.fromkeys(i for i in (x.strip() for x in ids.split(',')) if i))
        if len(wanted) > MAX_IDS:
            raise HTTPException(status_code=400, detail=f'at most {MAX_IDS} ids per request')
        found = await crops_by_ids(opensearch, wanted)
        return crops_page(total=len(found), page=1, page_size=len(wanted), crops=found)
    if limit is not None:
        page_size = limit
    guard_page_depth(page, page_size)
    try:
        listing = build_crop_list_query(
            sort=sort,
            item_filter_params=item_filter.model_dump(exclude_defaults=True),
            route_params={
                'cluster_id': cluster_id,
                'cluster_subid': cluster_subid,
                'label_source': label_source,
                'class_source': class_source,
                'label_validated': label_validated,
                'import_id': import_id,
                'dataset_split': dataset_split,
                'on_negative_frame': on_negative_frame,
                'proposed_by_import': proposed_by_import,
                'source': source,
                'needs_new_class': needs_new_class,
                'review_dismissed': review_dismissed,
                'min_blur_ratio': min_blur_ratio,
                'classifier_conf_lt': classifier_conf_lt,
                'item_text': item_text,
                'open_vocab_set': open_vocab_set,
                'source_prompt': source_prompt,
            },
            include_test=include_test,
            include_excluded=include_excluded,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = crop_list_body(listing, page=page, page_size=page_size)
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    # Computed orders (outliers / core_first / diverse) rank the matched
    # pool; None falls through to the plain sort above.
    ordered = await ordered_crops_page(
        opensearch,
        index=items_index(),
        order=order,
        query_clause=listing.query_clause,
        cluster_id=cluster_id,
        item_filter=listing.item_filter,
        page=page,
        page_size=page_size,
        k=k,
        n_pool=int(total),
        fetch_items=crops_by_ids,
    )
    if ordered is not None:
        return ordered

    hits = (resp.get('hits') or {}).get('hits') or []
    crops = [serialize_item(h.get('_source') or {}, h.get('_id', '')) for h in hits]
    return crops_page(total=int(total), page=page, page_size=page_size, crops=crops)


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
    try:
        # Still excludes the 1024-d embedding vectors (matching
        # every other item endpoint's behavior) -- but not
        # class_id_history, since a single-item view may legitimately
        # want it, unlike a paginated list.
        resp = await opensearch.get(
            index=items_index(), id=crop_id, _source_excludes=item_source_excludes()
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc
    src = (resp.get('_source') or {}) if isinstance(resp, dict) else {}
    return serialize_item(src, crop_id)

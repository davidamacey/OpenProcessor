"""Curation router sub-module: the image gallery and one image with its detections.

``GET /images`` pages the project's source images (filter, sort, served counts and boxes);
``GET /images/{image_id}`` is one image with every item on it. The pixels are served by
``images.py`` (`/images/{image_id}/file` and `/thumbnail`).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._image_view_models import ImageDetail
from src.routers.curation._item_filter_params import ItemFilterQuery
from src.schemas.curation.images import ImagePage
from src.services.curation import map_cells
from src.services.curation.image_detail import image_detail
from src.services.curation.image_view import (
    DEFAULT_SORT,
    LIMIT_DEFAULT,
    LIMIT_MAX,
    ImageQuery,
    ImageQueryError,
    LabelState,
    NearDupFilter,
    image_page,
)
from src.services.curation.location_policy_store import get_location_policy
from src.services.embedding_spaces.readers import item_write_fields


@router.get('/images', response_model=ImagePage, tags=['Images'])
async def list_images(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    source: Annotated[str | None, Query(description='Ingest source tag.')] = None,
    has_location: Annotated[
        bool | None, Query(description='true = a position is stored and may be served.')
    ] = None,
    geo_bbox: Annotated[
        str | None,
        Query(
            description='west,south,east,north in degrees; west > east crosses the antimeridian.'
        ),
    ] = None,
    in_holdout: Annotated[
        bool | None, Query(description='true = images with a held-out item; false = none.')
    ] = None,
    near_dup: Annotated[
        NearDupFilter | None,
        Query(
            description='member = in a group and not its representative; representative; '
            'none = in no group.'
        ),
    ] = None,
    label_state: Annotated[
        LabelState | None,
        Query(
            description='has_unlabeled, all_labeled (has items, none unlabeled) or no_items. '
            'Resolved by a join on the items.'
        ),
    ] = None,
    sort: Annotated[
        str, Query(description='One of the `sort_options` ids the response serves.')
    ] = DEFAULT_SORT,
    cursor: Annotated[
        str | None, Query(description='`next_cursor` of the previous page (same sort).')
    ] = None,
    limit: Annotated[int, Query(ge=1, le=LIMIT_MAX)] = LIMIT_DEFAULT,
    include_test: Annotated[
        bool, Query(description='Count and draw held-out test items too.')
    ] = False,
) -> ImagePage:
    """A page of the project's images. An image matches the item filter when at least one of its
    items does (`join_truncated`: the join used only the first 10,000 matching images). Each card
    carries its served counts and the boxes to draw; the page costs one query on the images index
    and one aggregation on the items index."""
    try:
        box = None if geo_bbox is None else map_cells.parse_bbox(geo_bbox)
        return await image_page(
            opensearch,
            ImageQuery(
                item_filter=item_filter,
                source=source,
                has_location=has_location,
                geo_box=box,
                in_holdout=in_holdout,
                near_dup=near_dup,
                label_state=label_state,
                sort=sort,
                cursor=cursor,
                limit=limit,
                include_test=include_test,
            ),
            await item_write_fields(),
            await get_location_policy(),
        )
    except (ImageQueryError, map_cells.MapQueryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    '/images/{image_id}',
    response_model=None,
    responses={200: {'model': ImageDetail}},
    tags=['Images'],
)
async def get_image(image_id: str, opensearch: OpenSearchDep) -> dict[str, Any]:
    """One image: its facts, the items detected on it (wire items, at most 500) and the boxes to
    draw, in image coordinates."""
    detail = await image_detail(opensearch, image_id, await get_location_policy())
    if detail is None:
        raise HTTPException(status_code=404, detail=f'image not found: {image_id}')
    return detail

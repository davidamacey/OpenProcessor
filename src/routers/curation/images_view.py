"""Curation router sub-module: the image gallery and one image with its detections.

``GET /images`` pages the project's source images (filter, sort, served counts and boxes);
``GET /images/{image_id}`` is one image with every item on it. The pixels are served by
``images.py`` (`/images/{image_id}/file` and `/thumbnail`).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._image_query_params import ImageViewQuery
from src.routers.curation._image_view_models import ImageDetail
from src.routers.curation._item_filter_params import ItemFilterQuery
from src.schemas.curation.images import ImagePage
from src.services.curation.image_detail import image_detail
from src.services.curation.image_view import (
    LIMIT_DEFAULT,
    LIMIT_MAX,
    ImageQuery,
    ImageQueryError,
    image_page,
)
from src.services.curation.list_cursor import CursorError
from src.services.curation.location_policy_store import get_location_policy
from src.services.embedding_spaces.readers import item_write_fields


@router.get('/images', response_model=ImagePage, tags=['Images'])
async def list_images(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    view: ImageViewQuery,
    cursor: Annotated[
        str | None, Query(description='`next_cursor` of the previous page (same sort).')
    ] = None,
    limit: Annotated[int, Query(ge=1, le=LIMIT_MAX)] = LIMIT_DEFAULT,
) -> ImagePage:
    """A page of the project's images. An image matches the item filter when at least one of its
    items does (`join_truncated`: the join used only the first 10,000 matching images). Each card
    carries its served counts and the boxes to draw; the page costs one query on the images index
    and one aggregation on the items index."""
    try:
        return await image_page(
            opensearch,
            ImageQuery(item_filter=item_filter, view=view, cursor=cursor, limit=limit),
            await item_write_fields(),
            await get_location_policy(),
        )
    except CursorError as exc:
        raise api_error(400, 'cursor_invalid', str(exc)) from exc
    except ImageQueryError as exc:
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

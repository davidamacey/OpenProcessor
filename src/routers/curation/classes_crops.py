"""Curation router sub-module — a class's crops."""

from __future__ import annotations

from typing import Any

from fastapi import Query

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._crop_params import CropBrowse
from src.routers.curation._cursor_params import CursorQ
from src.routers.curation._item_models import CropsPageResponse
from src.routers.curation.crops_list import list_crops
from src.services.curation.item_filter import ItemFilter


@router.get(
    '/classes/{class_id}/crops',
    response_model=None,
    responses={200: {'model': CropsPageResponse}},
)
async def class_crops(
    class_id: int,
    opensearch: OpenSearchDep,
    cursor: CursorQ = None,
    limit: int = Query(50, ge=1, le=500),
    include_test: bool = False,
) -> dict[str, Any]:
    """Crops in a single class."""
    return await list_crops(
        opensearch=opensearch,
        item_filter=ItemFilter(class_id=class_id),
        browse=CropBrowse(include_test=include_test),
        cursor=cursor,
        limit=limit,
    )

"""Curation router sub-module — a class's crops and the registry-to-index sync."""

from __future__ import annotations

from typing import Any

from fastapi import Query

from src.routers.curation._common import (
    CropsPageResponse,
    OpenSearchDep,
    get_class_registry,
    router,
)
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
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    include_test: bool = False,
) -> dict[str, Any]:
    """Crops in a single class."""
    return await list_crops(
        opensearch=opensearch,
        page=page,
        page_size=page_size,
        item_filter=ItemFilter(class_id=class_id),
        include_test=include_test,
    )


@router.post('/classes/sync_to_opensearch')
async def sync_classes(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Rebuild the classes index from the on-disk registry."""
    return await get_class_registry().sync_to_opensearch(opensearch)

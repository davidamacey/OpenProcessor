"""Curation router sub-module: ``GET /detections/summary``."""

from __future__ import annotations

from fastapi import HTTPException

from src.routers.curation._common import OpenSearchDep, items_index, router
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.services.curation.detections_summary import DetectionsSummary, detections_summary
from src.services.curation.index_bootstrap import ensure_indexes


@router.get('/detections/summary', response_model=DetectionsSummary)
async def get_detections_summary(
    opensearch: OpenSearchDep, item_filter: ItemFilterQuery
) -> DetectionsSummary:
    """Stored detections per detector label, each with its embedding breakdown
    (embedded, not embedded and why), over the items the shared item filter
    selects (every item when no filter parameter is set), plus a
    ``suggested_reprocess`` body that embeds the missing ones."""
    await ensure_indexes(opensearch)
    try:
        return await detections_summary(opensearch, items_index(), item_filter)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

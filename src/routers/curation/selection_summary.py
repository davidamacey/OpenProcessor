"""Curation router sub-module: what a bulk action on a selection would touch."""

from __future__ import annotations

from fastapi import HTTPException

from src.routers.curation._common import OpenSearchDep, router
from src.schemas.curation.selection import SelectionSummary
from src.services.curation.item_selection import ItemSelection, SelectionError
from src.services.curation.location_policy_store import get_location_policy
from src.services.curation.selection_summary import selection_summary
from src.services.embedding_spaces.readers import item_write_fields


@router.post('/selection/summary', response_model=SelectionSummary, tags=['Selection'])
async def post_selection_summary(
    body: ItemSelection, opensearch: OpenSearchDep
) -> SelectionSummary:
    """Detections, images, clusters, locks and classes of a selection, counted in one query on
    the items index. Refused with 422 when the selection names more than the bulk-write cap."""
    try:
        return await selection_summary(
            opensearch, body, await item_write_fields(), await get_location_policy()
        )
    except SelectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

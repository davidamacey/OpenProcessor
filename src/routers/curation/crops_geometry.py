"""Curation router sub-module: move or resize an item's box, and undo it."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._item_write_models import GeometryWritten
from src.routers.curation.search import _get_pe_encoder
from src.schemas.curation.config_common import ApiErrorResponse
from src.schemas.curation.images import GeometryPatchRequest
from src.services.curation.image_detail import image_boxes_and_counts
from src.services.curation.item_geometry import GeometryResult, edit_geometry, undo_geometry
from src.services.curation.wire import serialize_item
from src.services.embedding_spaces.readers import item_state_fields


_ERRORS: dict[int | str, dict[str, Any]] = {
    404: {'model': ApiErrorResponse},
    409: {'model': ApiErrorResponse},
    422: {'model': ApiErrorResponse},
}


async def _written(opensearch: Any, crop_id: str, result: GeometryResult) -> dict[str, Any]:
    boxes, _counts = await image_boxes_and_counts(
        opensearch, str(result.source.get('image_id') or '')
    )
    return {
        'item': serialize_item(result.source, crop_id, state_fields=await item_state_fields()),
        'image_boxes': [b.model_dump(mode='json') for b in boxes],
        'embedding_state': result.embed.state,
        'embedding_message': result.embed.message,
    }


@router.patch(
    '/crops/{crop_id}/geometry',
    response_model=None,
    responses={200: {'model': GeometryWritten}, **_ERRORS},
    tags=['Crops'],
)
async def patch_geometry(
    crop_id: str, payload: GeometryPatchRequest, request: Request, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Move or resize an item's box. `expected_geometry_revision` is the revision the editor
    holds (0 for an item never edited); a stale one is `409 geometry_conflict` carrying the
    current item. Region boxes the new box would leave outside it are `422 regions_outside_item`
    until they are listed in `drop_region_box_ids`. The old vector is dropped and the item
    re-embedded in this request."""
    result = await edit_geometry(
        opensearch,
        _get_pe_encoder(request),
        crop_id,
        bbox=payload.bbox_norm,
        expected_revision=payload.expected_geometry_revision,
        drop_region_box_ids=payload.drop_region_box_ids,
    )
    return await _written(opensearch, crop_id, result)


@router.post(
    '/crops/{crop_id}/geometry/undo',
    response_model=None,
    responses={200: {'model': GeometryWritten}, **_ERRORS},
    tags=['Crops'],
)
async def undo_crop_geometry(
    crop_id: str, request: Request, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Put the box back to what it was before the newest edit that is not already undone.
    `409 nothing_to_undo` when none is left; an undo of an undo is not restorable."""
    result = await undo_geometry(opensearch, _get_pe_encoder(request), crop_id)
    return await _written(opensearch, crop_id, result)

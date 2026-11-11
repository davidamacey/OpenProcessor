"""Curation router sub-module: draw a box on an image and it becomes an item."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._item_write_models import ItemCreated
from src.routers.curation.search import _get_pe_encoder
from src.schemas.curation.config_common import ApiErrorResponse
from src.schemas.curation.images import ItemCreateRequest
from src.services.curation.image_detail import image_boxes_and_counts
from src.services.curation.image_item_write import create_drawn_item
from src.services.curation.wire import serialize_item
from src.services.embedding_spaces.readers import item_state_fields


@router.post(
    '/images/{image_id}/items',
    response_model=None,
    status_code=201,
    responses={
        200: {'model': ItemCreated, 'description': 'A repeat of the same `request_id`.'},
        201: {'model': ItemCreated},
        404: {'model': ApiErrorResponse},
        422: {'model': ApiErrorResponse},
    },
    tags=['Images'],
)
async def create_item(
    image_id: str,
    payload: ItemCreateRequest,
    request: Request,
    opensearch: OpenSearchDep,
) -> dict[str, Any] | JSONResponse:
    """Create an item from a box a person drew. The class is resolved by name (an unknown or
    deprecated one is `unknown_class`); the item carries a human class and human geometry, is held
    out when the image is, and is embedded in this request (a failure keeps the item and serves
    `embedding_state: deferred` with a message). Sending the same `request_id` again returns the
    first item with `created: false`."""
    drawn = await create_drawn_item(
        opensearch,
        _get_pe_encoder(request),
        image_id=image_id,
        bbox=payload.bbox_norm,
        class_name=payload.class_name,
        request_id=payload.request_id,
    )
    boxes, counts = await image_boxes_and_counts(opensearch, image_id)
    body = {
        'item': serialize_item(drawn.source, state_fields=await item_state_fields()),
        'created': drawn.created,
        'image_boxes': [b.model_dump(mode='json') for b in boxes],
        'image_counts': counts.model_dump(mode='json'),
        'embedding_state': drawn.embed.state,
        'embedding_message': drawn.embed.message,
    }
    return body if drawn.created else JSONResponse(body, status_code=200)

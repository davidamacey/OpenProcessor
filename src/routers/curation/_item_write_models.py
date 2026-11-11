"""Response models of the annotation writes: a drawn item and a box edit.

Documentation models: the handlers return plain dicts built from the wire serializer, so a stored
value of an unexpected type never 500s a write that has already stood.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.embedding_state import EmbeddingState
from src.routers.curation._item_models import ItemDoc
from src.schemas.curation.images import BoxView, ImageCounts


class ItemCreated(BaseModel):
    item: ItemDoc
    created: bool = Field(description='False on an idempotent repeat of the same `request_id`.')
    image_boxes: list[BoxView]
    image_counts: ImageCounts
    embedding_state: EmbeddingState
    embedding_message: str | None = Field(
        default=None, description='Served text when the item is deferred; null otherwise.'
    )


class GeometryWritten(BaseModel):
    item: ItemDoc
    image_boxes: list[BoxView]
    embedding_state: EmbeddingState
    embedding_message: str | None = None

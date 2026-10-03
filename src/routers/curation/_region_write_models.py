"""Wire models for the region write routes (box edits and whole-set statuses).

Documentation/OpenAPI models: routes declare them through ``responses=``.
Leaf module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.routers.curation._item_models import ItemDoc  # noqa: TC001 - pydantic field type
from src.routers.curation._region_row_models import RegionRow  # noqa: TC001 - pydantic


class VectorRefresh(BaseModel):
    """What the write did to the touched boxes' vectors: boxes embedded now,
    and boxes still without a valid vector (encoder unavailable, image
    unreadable or the encoder raised): retry with a reprocess ``embed``."""

    embedded: int
    pending: int


class RegionItemWriteResponse(BaseModel):
    """``PUT /crops/{crop_id}/regions``."""

    crop_id: str
    item: ItemDoc
    vector_refresh: VectorRefresh


class RegionMetaPatchResponse(RegionItemWriteResponse):
    """``PATCH /crops/{crop_id}/region_meta``."""

    updated_fields: list[str]


class RegionBoxPatchResponse(RegionItemWriteResponse):
    """``PATCH /crops/{crop_id}/regions/{box_id}``."""

    box_id: str


class RegionBatchWriteResponse(BaseModel):
    """``POST /regions/batch_status``, ``PUT /crops/batch_regions`` and
    ``POST /regions/batch_box_state``: ``items`` are rows (the post-write wire
    item, with ``region_box_id`` null for a whole-set write)."""

    updated: int
    conflicts: list[dict[str, Any]]
    invalid: list[dict[str, Any]]
    items: list[RegionRow]
    vector_refresh: VectorRefresh = Field(
        description='Zero counts when the batch named nothing to write.'
    )


__all__ = [
    'RegionBatchWriteResponse',
    'RegionBoxPatchResponse',
    'RegionItemWriteResponse',
    'RegionMetaPatchResponse',
    'VectorRefresh',
]

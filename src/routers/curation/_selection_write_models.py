"""Wire models for the bulk item writes that take explicit ids or a selection
(label, move, exclude, unexclude). Documentation/OpenAPI models: routes declare
them through ``responses=``. Leaf module."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SelectionDryRunResponse(BaseModel):
    """What a ``dry_run`` request returns: the count the write would change, and nothing written."""

    dry_run: Literal[True]
    selected: int


class BatchConflict(BaseModel):
    crop_id: str
    current_source: str | None = Field(description='The class_source that won the race.')


class BatchRelabelResponse(BaseModel):
    """``PUT /crops/batch_label`` and ``POST /crops/move``."""

    updated: int
    updated_ids: list[str] = Field(description='The items actually written (the undo target).')
    conflicts: list[BatchConflict]


class BatchExcludeResponse(BaseModel):
    """``POST /crops/batch_exclude``."""

    excluded: int
    updated_ids: list[str] = Field(description='The items actually excluded (the undo target).')
    errors: int


class BatchUnexcludeResponse(BaseModel):
    """``POST /crops/batch_unexclude``."""

    unexcluded: int
    updated_ids: list[str] = Field(description='The items actually restored.')
    errors: int


__all__ = [
    'BatchConflict',
    'BatchExcludeResponse',
    'BatchRelabelResponse',
    'BatchUnexcludeResponse',
    'SelectionDryRunResponse',
]

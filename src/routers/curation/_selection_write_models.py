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


class LabelOpAccepted(BaseModel):
    """``PUT /crops/batch_label``, ``POST /crops/move`` and a real undo: the job is queued (202)."""

    op_id: str = Field(description='The recorded operation (undo it by id).')
    job_id: str = Field(description='The queued job; poll GET /jobs/{job_id}.')
    status: Literal['queued'] = 'queued'


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
    'BatchExcludeResponse',
    'BatchUnexcludeResponse',
    'LabelOpAccepted',
    'SelectionDryRunResponse',
]

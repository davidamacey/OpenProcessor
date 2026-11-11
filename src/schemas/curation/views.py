"""Response models shared by the item list routes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HiddenCounts(BaseModel):
    """Items a list leaves out, counted over the items that match the request's filter."""

    test_holdout: int = Field(
        description='Items matching the filter that the frozen test holdout hides (0 when included).'
    )
    excluded: int = Field(
        description='Items matching the filter that are ignored (0 when included).'
    )

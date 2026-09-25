"""Response model for ``GET {prefix}/review/tabs``.

Typed so the OpenAPI contract (``contracts/openapi/curation.json``) carries
the shape a client renders from; the values come from
:func:`src.services.curation.review_queries.review_tab_catalog`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReviewFilterOption(BaseModel):
    value: str
    label: str


class ReviewFilterSpec(BaseModel):
    param: str = Field(description='Query parameter on GET /review/{tab} and its locate route.')
    kind: Literal['enum']
    label: str
    options: list[ReviewFilterOption]


class ReviewTab(BaseModel):
    id: str
    label: str
    description: str
    filters: list[str] = Field(description='Query parameters the tab honours.')
    filter_defaults: dict[str, Any] = Field(
        description='Value the tab applies when a parameter is omitted.'
    )
    filter_specs: list[ReviewFilterSpec] = Field(
        description='Self-describing spec for each honoured filter with a fixed value set.'
    )


class ReviewEmptyState(BaseModel):
    """C3: underlying-state flags a client uses to word ANY tab's
    zero-result state without a per-tab round trip -- the same signals
    ``GET /review/{tab}``'s own ``empty_reason`` is derived from."""

    has_probe_predictions: bool
    has_item_scores: bool


class ReviewTabsResponse(BaseModel):
    tabs: list[ReviewTab]
    empty_state: ReviewEmptyState


__all__ = [
    'ReviewEmptyState',
    'ReviewFilterOption',
    'ReviewFilterSpec',
    'ReviewTab',
    'ReviewTabsResponse',
]

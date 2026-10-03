"""Response model for ``GET {prefix}/review/tabs``.

Typed so the OpenAPI contract (``contracts/openapi/curation.json``) carries
the shape a client renders from; the values come from
:func:`src.services.curation.review_queries.review_tab_catalog`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.services.curation.reprocess_models import ReprocessRequest  # noqa: TC001 - pydantic


ReviewFilterKind = Literal['enum', 'multi_enum', 'class_names', 'bool', 'number', 'integer', 'text']


class ReviewFilterOption(BaseModel):
    value: str
    label: str


class ReviewFilterSpec(BaseModel):
    param: str = Field(description='Query parameter on GET /review/{tab} and its locate route.')
    kind: ReviewFilterKind
    label: str
    options: list[ReviewFilterOption] = Field(
        description='The fixed values of an enum / multi_enum filter; empty for every other kind.'
    )
    min: float | None = Field(description='Lower bound of a number / integer filter.')
    max: float | None = Field(description='Upper bound of a number / integer filter.')
    description: str = Field(description='How to fill the filter; empty when the label says it.')
    default: Any = Field(
        description=(
            'Value applied when the parameter is omitted, read off the filter model: '
            'null = no filter, [] = none selected, false = off.'
        )
    )
    allows_unset: bool = Field(
        description=(
            'True when the default is null: omit the parameter for "no filter" '
            '(an enum then needs a client-side "Any" choice; never send an empty string).'
        )
    )


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
    """Underlying-state flags a client uses to word ANY tab's
    zero-result state without a per-tab round trip -- the same signals
    ``GET /review/{tab}``'s own ``empty_reason`` is derived from."""

    has_probe_predictions: bool
    has_item_scores: bool
    has_imported_labels: bool
    has_unembedded_items: bool = Field(
        description='Some item in the project has no vector, so a queue that skips them can be empty.'
    )
    suggested_reprocess: ReprocessRequest | None = Field(
        description=(
            'The embed request to POST to /reprocess (dry run first) when '
            'has_unembedded_items; null otherwise.'
        )
    )


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

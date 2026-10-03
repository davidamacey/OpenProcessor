"""Wire models for the item pages other than ``GET /crops``: the review queue,
its locate twin and the semantic search.

Documentation/OpenAPI models only (routes declare them through ``responses=``):
handlers return the serialized items directly, so a stored value of an
unexpected type never 500s a page. Leaf module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.routers.curation._item_models import ItemDoc


class ReviewItemDoc(ItemDoc):
    reason: str = Field(description="Why the item is in this tab's queue.")


class ReviewQueuePage(BaseModel):
    """``GET /review/{tab}``."""

    total: int
    page: int
    page_size: int
    items: list[ReviewItemDoc]
    sort_applied: str
    sort_fallback_reason: str | None = Field(
        description="Set when the default sort's field has no coverage and sort_applied is its fallback."
    )
    empty_reason: str | None = Field(
        description='Why a zero-result queue is empty (probe never run, scores never computed, ...); null otherwise.'
    )


class ReviewLocateResponse(BaseModel):
    """``GET /review/{tab}/locate``: where an item sits in the queue the same
    filters and sort would serve. ``rank`` is 0-based; ``page`` is 1-based."""

    crop_id: str
    in_queue: bool
    rank: int | None
    page: int | None
    page_size: int
    total: int | None
    reason: Literal['not_found', 'filtered_out', 'region_profile_off'] | None = Field(
        description='Why it is not in the queue; null when in_queue.'
    )
    sort_applied: str
    sort_fallback_reason: str | None


class SearchItemDoc(ItemDoc):
    semantic_score: float | None = Field(description='Raw kNN similarity of the item to the query.')


class SearchTextResponse(BaseModel):
    """``GET /search/text``."""

    items: list[SearchItemDoc]
    total: int
    page: int
    page_size: int
    unembedded_in_scope: int = Field(
        description=(
            'In-scope items the search cannot see because they have no vector: '
            'an empty result with a non-zero count is not "no matches".'
        )
    )


__all__ = ['ReviewItemDoc', 'ReviewLocateResponse', 'ReviewQueuePage', 'SearchTextResponse']

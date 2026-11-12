"""The review queue's filter model, a leaf both the request builder and the
filter specs read (the specs take their defaults from it)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.curation.item_filter import ItemFilter


if TYPE_CHECKING:
    from src.services.curation.region_text_search import TextMatch

REGION_STATUS_DEFAULT = 'all'


@dataclass(frozen=True)
class ReviewFilters:
    include_test: bool = False
    text: str | None = None
    max_rank: int | None = None
    min_blur_ratio: float | None = None
    min_mistakenness: float | None = None
    hide_near_duplicates: bool = False
    source: str | None = None
    # The shared item filter (class names, confidence band, size, origin,
    # embedding state, review status), minus ``max_rank``, which a tab can
    # default and so is read above.
    item: ItemFilter = dataclasses.field(default_factory=ItemFilter)
    region_status: str | None = REGION_STATUS_DEFAULT
    region_text_match: TextMatch | None = None
    combine_conflict: bool = False
    on_negative_frame: bool | None = None

"""Wire models of the image view: `GET .../images`, `.../images/{image_id}` and the boxes they draw.

An image page is a view model: every number, label, box and empty reason the page shows is served
here, so the browser computes none of them.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.curation.views import HiddenCounts, SortOption


BoxTone = Literal['validated', 'machine', 'unlabeled', 'holdout', 'rejected']


class BoxView(BaseModel):
    """One box to draw over an image, in image coordinates."""

    crop_id: str = Field(description='The item the box belongs to.')
    box_id: str | None = Field(
        default=None, description='The region box id; null for the item box itself.'
    )
    kind: Literal['item', 'region']
    bbox_norm: tuple[float, float, float, float] = Field(
        description='[x1, y1, x2, y2], normalized 0..1 in the source image.'
    )
    label: str = Field(description='The text to show on the box: the class name, or "unlabeled".')
    class_name: str | None = None
    tone: BoxTone = Field(description='How the box is drawn: its review state, served.')
    state: str | None = Field(default=None, description='The region box state; null for an item.')


class ImageCounts(BaseModel):
    """The items on one image, as the default view counts them (excluded items never count;
    held-out items count only when the request asked for them)."""

    items: int
    labeled: int
    validated: int
    unlabeled: int
    in_holdout: int = Field(description='Held-out items on the image, whatever the request asked.')


class ImageNearDup(BaseModel):
    group_id: str
    purpose: Literal['dedup', 'leak']
    size: int | None = None
    is_representative: bool | None = None


class ImageCard(BaseModel):
    image_id: str
    file_name: str
    thumbnail_url: str
    width: int | None = None
    height: int | None = None
    source: str = ''
    indexed_at: str | None = None
    taken_at: str | None = Field(
        default=None, description='The capture time (UTC when the offset is known, else local).'
    )
    has_location: bool = Field(
        description='A position is stored and the location policy lets it be served.'
    )
    counts: ImageCounts
    near_dup: ImageNearDup | None = None
    boxes: list[BoxView]
    boxes_truncated: bool = Field(description='The image has more items than `boxes` draws.')


EmptyReason = Literal['no_images', 'no_match']


class ImagePage(BaseModel):
    items: list[ImageCard]
    next_cursor: str | None = Field(
        default=None, description='Opaque; pass back as `cursor`. Null on the last page.'
    )
    total: int
    join_truncated: bool = Field(
        description='An item-level filter matched more images than the join could use.'
    )
    sort_applied: str
    sort_options: list[SortOption]
    hidden: HiddenCounts
    empty_reason: EmptyReason | None = Field(
        default=None, description='Why `items` is empty; null when it is not.'
    )


class ImageLocation(BaseModel):
    latitude: float
    longitude: float
    precision: str | None = None


class ImageInfo(BaseModel):
    image_id: str
    file_name: str
    file_url: str = Field(description='The clean source pixels (`GET .../images/{id}/file`).')
    width: int | None = None
    height: int | None = None
    source: str = ''
    indexed_at: str | None = None
    taken_at: str | None = None
    location: ImageLocation | None = Field(
        default=None, description="Served under the project's location policy; null when dropped."
    )
    dataset_split: str | None = None
    in_holdout: bool
    near_dup: ImageNearDup | None = None


class ItemCreateRequest(BaseModel):
    """Draw a box: it becomes an item with a person's class and a person's geometry."""

    model_config = ConfigDict(extra='forbid')

    bbox_norm: tuple[float, float, float, float] = Field(
        description='[x1, y1, x2, y2] in image coordinates, each in [0, 1], x1 < x2 and y1 < y2.'
    )
    class_name: str = Field(min_length=1, max_length=128, description='An active registry class.')
    request_id: uuid.UUID = Field(
        description='Sent again unchanged, it returns the first item instead of a second.'
    )


class GeometryPatchRequest(BaseModel):
    """Move or resize an item's box."""

    model_config = ConfigDict(extra='forbid')

    bbox_norm: tuple[float, float, float, float]
    expected_geometry_revision: int = Field(
        ge=0, description='The `geometry_revision` the editor holds; 0 for an item never edited.'
    )
    drop_region_box_ids: list[str] = Field(
        default_factory=list,
        description='Region boxes that fall outside the new box and may be deleted with it.',
    )

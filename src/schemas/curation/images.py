"""Wire models of the image view: `GET .../images`, `.../images/{image_id}` and the boxes they draw.

An image page is a view model: every number, label, box and empty reason the page shows is served
here, so the browser computes none of them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.curation.views import HiddenCounts, SortOption


BoxLabelKind = Literal['class', 'proposed', 'detector', 'region', 'none']
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
    label: str = Field(
        description='The text to show on the box: the item display name or the region text.'
    )
    label_kind: BoxLabelKind = Field(description='Where `label` came from.')
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


class ImageLocate(BaseModel):
    """``GET /images/locate``: where an image sits in the list the same filters and sort serve."""

    image_id: str
    in_view: bool
    index: int | None = Field(description='0-based position in the list; null when not in it.')
    total: int = Field(description='Images in the list.')
    cursor_before: str | None = Field(
        description='Pass as `cursor` to `GET /images` for the page that starts at this image; '
        'null for the first row and when not in the list.'
    )
    cursor_after: str | None = Field(
        description='Pass as `cursor` for the page right after this image; null when not in it.'
    )
    reason: Literal['not_found', 'filtered_out'] | None = Field(
        description='Why it is not in the list; null when in_view.'
    )
    reason_message: str | None = Field(description='The served sentence for `reason`.')

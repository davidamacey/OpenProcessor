"""Documentation model of `GET .../images/{image_id}` (the handler returns a plain dict, so a stored
value of an unexpected type never 500s; a test pins the keys)."""

from __future__ import annotations

from pydantic import BaseModel

from src.routers.curation._item_models import ItemDoc
from src.schemas.curation.images import BoxView, ImageCounts, ImageInfo


class ImageDetail(BaseModel):
    """One image and everything detected on it. `items` are wire items ordered by
    `crop_rank_in_image` (at most 500, `items_truncated` says when there were more); `boxes` draws
    all of them in image coordinates."""

    image: ImageInfo
    counts: ImageCounts
    items: list[ItemDoc]
    boxes: list[BoxView]
    items_truncated: bool

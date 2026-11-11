"""Read the served facts of one images-index document: name, capture time, position, near-dup
group and the URLs of its pixels. Shared by the gallery card and the single-image view."""

from __future__ import annotations

import posixpath
from typing import Any

from src.clients.curation_opensearch.capture_fields import block_within_policy
from src.schemas.curation.images import ImageLocation, ImageNearDup


#: The images-index fields both views read.
IMAGE_SOURCE_FIELDS: tuple[str, ...] = (
    'image_id',
    'image_path',
    'width',
    'height',
    'source',
    'indexed_at',
    'capture',
    'dataset_split',
    'near_dup_group_id',
    'near_dup_leak_group_id',
    'near_dup_is_representative',
    'near_dup_group_size',
)


def file_name(src: dict[str, Any]) -> str:
    return posixpath.basename(str(src.get('image_path') or '').replace('\\', '/'))


def served_capture(src: dict[str, Any], policy: str) -> dict[str, Any]:
    """The stored ``capture`` block as the project's location policy lets it be served."""
    block = src.get('capture')
    return block_within_policy(block, policy) if isinstance(block, dict) else {}


def taken_at(capture: dict[str, Any]) -> str | None:
    value = capture.get('taken_utc') or capture.get('taken_local')
    return None if value is None else str(value)


def location_of(capture: dict[str, Any]) -> ImageLocation | None:
    loc = capture.get('location')
    if not isinstance(loc, dict) or 'lat' not in loc or 'lon' not in loc:
        return None
    precision = capture.get('location_precision')
    return ImageLocation(
        latitude=float(loc['lat']),
        longitude=float(loc['lon']),
        precision=None if precision is None else str(precision),
    )


def near_dup_of(src: dict[str, Any]) -> ImageNearDup | None:
    """The image's dedup group, else its leak group; ``None`` when it is in neither."""
    if group := src.get('near_dup_group_id'):
        return ImageNearDup(
            group_id=str(group),
            purpose='dedup',
            size=src.get('near_dup_group_size'),
            is_representative=src.get('near_dup_is_representative'),
        )
    if leak := src.get('near_dup_leak_group_id'):
        return ImageNearDup(group_id=str(leak), purpose='leak')
    return None


def thumbnail_url(prefix: str, image_id: str) -> str:
    return f'{prefix}/images/{image_id}/thumbnail'


def file_url(prefix: str, image_id: str) -> str:
    return f'{prefix}/images/{image_id}/file'

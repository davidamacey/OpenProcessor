"""One image with every item detected on it: `GET .../images/{image_id}` as a view model."""

from __future__ import annotations

import asyncio
from typing import Any, Final

from opensearchpy.exceptions import NotFoundError
from PIL import Image

from src.config.curation_accessors import images_index, items_index
from src.config.project_context import project_api_base
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.schemas.curation.images import BoxView, ImageCounts, ImageInfo
from src.services.curation.image_boxes import boxes_of
from src.services.curation.image_counts import item_facts
from src.services.curation.image_facts import (
    IMAGE_SOURCE_FIELDS,
    file_name,
    file_url,
    location_of,
    near_dup_of,
    served_capture,
    taken_at,
)
from src.services.curation.image_serving import resolve_crop_root, resolve_safe_path
from src.services.curation.wire import item_context_source_excludes, serialize_item
from src.services.embedding_spaces.readers import item_state_fields, vector_source_excludes
from src.utils.image_decode import IMAGE_DECODE_ERRORS


logger = get_logger(__name__)

ITEMS_MAX: Final = 500
# An unservable path or anything Pillow raises on a file it cannot read.
_UNREADABLE_IMAGE: tuple[type[Exception], ...] = (RequestRejectedError, *IMAGE_DECODE_ERRORS)


def _pixel_size_from_header(image_path: str) -> tuple[int, int] | None:
    """Best-effort ``(width, height)`` from the image file header, for an images-index document
    that predates width and height. ``None`` on any resolution or decode failure."""
    try:
        safe_path = resolve_safe_path(image_path, resolve_crop_root(image_path))
        with Image.open(safe_path) as img:
            return img.size
    except _UNREADABLE_IMAGE as exc:
        logger.debug('image_pixel_size_failed', image_path=image_path, error=str(exc))
        return None


async def image_source(opensearch: Any, image_id: str, fields: list[str]) -> dict[str, Any] | None:
    """The bound project's images-index document ``image_id`` (just ``fields``); ``None`` when it
    holds no such image."""
    try:
        resp = await opensearch.get(index=images_index(), id=image_id, _source_includes=fields)
    except NotFoundError:
        return None
    if not resp.get('found', True):
        return None
    return resp.get('_source') or {}


async def image_detail(opensearch: Any, image_id: str, policy: str) -> dict[str, Any] | None:
    """``{image, counts, items, boxes, items_truncated}`` or ``None`` for an unknown image.

    Items are wire items ordered by ``crop_rank_in_image``, at most 500; ``boxes`` draws every
    one of them (excluded and held-out items included, each with its own tone)."""
    src = await image_source(opensearch, image_id, list(IMAGE_SOURCE_FIELDS))
    if src is None:
        return None
    width, height = src.get('width'), src.get('height')
    if (width is None or height is None) and src.get('image_path'):
        size = await asyncio.to_thread(_pixel_size_from_header, str(src['image_path']))
        if size is not None:
            width, height = size
    vector_excludes = await vector_source_excludes()
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': ITEMS_MAX,
            'track_total_hits': True,
            'query': {'term': {'image_id': image_id}},
            'sort': [{'crop_rank_in_image': {'order': 'asc', 'missing': '_last'}}],
            '_source': {'excludes': item_context_source_excludes(vector_excludes)},
        },
    )
    hits = resp['hits']['hits']
    state_fields = await item_state_fields()
    items = [
        serialize_item(h.get('_source') or {}, h.get('_id', ''), state_fields=state_fields)
        for h in hits
    ]
    boxes: list[BoxView] = boxes_of((str(h.get('_id') or ''), h.get('_source') or {}) for h in hits)
    counts: ImageCounts = (
        await item_facts(opensearch, [image_id], include_test=True, box_limit=0)
    )[image_id].counts
    capture = served_capture(src, policy)
    info = ImageInfo(
        image_id=image_id,
        file_name=file_name(src),
        file_url=file_url(project_api_base(), image_id),
        width=width,
        height=height,
        source=src.get('source') or '',
        indexed_at=src.get('indexed_at'),
        taken_at=taken_at(capture),
        location=location_of(capture),
        dataset_split=src.get('dataset_split'),
        in_holdout=counts.in_holdout > 0,
        near_dup=near_dup_of(src),
    )
    return {
        'image': info.model_dump(mode='json'),
        'counts': counts.model_dump(mode='json'),
        'items': items,
        'boxes': [b.model_dump(mode='json') for b in boxes],
        'items_truncated': int(resp['hits']['total']['value']) > len(hits),
    }

"""The item-crop JPEG a worker (and a test run) sees, from one function.

Cache first, source image second: the ingest-time RAM crop cache
(``<crop_cache_dir>/<crop_id>.jpg``) when it has the crop, else open the
source image, EXIF-transpose, crop and re-encode. The detection worker's
``_crop_jpeg_for_task`` delegates here, and so do the test-on-crop routes
(W5), so a preview crop is byte-for-byte the crop the worker would send to
its detector and VLM.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import get_curation_config
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger('crop_bytes')

CROP_JPEG_QUALITY = 90

# Process-local counters, reset on restart; the worker's periodic metrics
# log reports them.
_cache_hits = 0
_cache_misses = 0


def cache_stats() -> tuple[int, int]:
    """``(hits, misses)`` of the crop cache since this process started."""
    return _cache_hits, _cache_misses


def crop_jpeg_from_cache(crop_id: str) -> bytes | None:
    """The cached item-crop JPEG bytes, or ``None`` on a miss."""
    global _cache_hits, _cache_misses  # noqa: PLW0603 - counters are intentionally module-level
    cache_dir = get_curation_config().crop_cache_dir
    if not cache_dir:
        _cache_misses += 1
        return None
    path = Path(cache_dir) / f'{crop_id}.jpg'
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        _cache_misses += 1
        return None
    except OSError as exc:
        _cache_misses += 1
        logger.warning('crop_cache_read_error', crop_id=crop_id, error=str(exc))
        return None
    _cache_hits += 1
    return data


def crop_jpeg_from_disk(image_path: str, bbox: tuple[float, float, float, float]) -> bytes | None:
    """A JPEG of the item crop cut from the source image on disk.

    The slow path: re-opens the source, EXIF-transposes, crops and
    re-encodes. ``None`` when the image is missing or unreadable.
    """
    p = Path(image_path)
    if not p.is_file():
        logger.warning('image_missing', path=image_path)
        return None
    try:
        with p.open('rb') as f:
            img = Image.open(f)
            img.load()
            img = ImageOps.exif_transpose(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
    except UnidentifiedImageError:
        logger.warning('image_unreadable', path=image_path)
        return None
    except OSError as exc:
        logger.warning('image_io_error', path=image_path, error=str(exc))
        return None

    full_w, full_h = img.size
    x1, y1, x2, y2 = bbox
    x1i = max(0, round(x1 * full_w))
    y1i = max(0, round(y1 * full_h))
    x2i = max(x1i + 1, round(x2 * full_w))
    y2i = max(y1i + 1, round(y2 * full_h))
    if x2i <= x1i or y2i <= y1i:
        return None
    crop = img.crop((x1i, y1i, x2i, y2i))
    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=CROP_JPEG_QUALITY)
    return buf.getvalue()


def load_item_crop_jpeg(
    crop_id: str, image_path: str, bbox: tuple[float, float, float, float]
) -> bytes | None:
    """The item crop's JPEG: the crop cache when it has ``crop_id``, else the
    source image. ``image_path`` must already be resolved to a readable path
    (the API resolves it under the crop root first)."""
    cached = crop_jpeg_from_cache(crop_id)
    if cached is not None:
        return cached
    return crop_jpeg_from_disk(image_path, bbox)


# --- API-side loaders (the stored image path is resolved under the crop root) ---

#: Long edge of the crops sent to a VLM by the label and region routes.
VLM_CROP_SIZE = 224


def resolved_image_path(image_path: str) -> Path:
    """``image_path`` resolved (and traversal-checked) under its crop root.

    Raises ``HTTPException`` (400/404) when the stored path is not servable.
    """
    from src.services.curation.image_serving import resolve_crop_root, resolve_safe_path

    return resolve_safe_path(image_path, resolve_crop_root(image_path))


def load_vlm_item_jpeg(
    crop_id: str,
    image_path: str,
    bbox: Sequence[float],
    *,
    cache_dir: str | Path | None,
) -> bytes | None:
    """The item crop at VLM size: the ingest-time crop cache (``cache_dir``)
    when it has the crop (shrunk to :data:`VLM_CROP_SIZE`), else a thumbnail
    cut from the source image. ``None`` when neither can be read."""
    from src.services.curation.image_serving import THUMBNAIL_CACHE

    if cache_dir:
        try:
            img = Image.open(io.BytesIO((Path(cache_dir) / f'{crop_id}.jpg').read_bytes()))
            img.thumbnail((VLM_CROP_SIZE, VLM_CROP_SIZE))
            buf = io.BytesIO()
            img.convert('RGB').save(buf, format='JPEG', quality=CROP_JPEG_QUALITY)
            return buf.getvalue()
        except FileNotFoundError:
            pass  # fall through to the source image
        except Exception as exc:
            logger.warning('vlm_crop_cache_read_failed', crop_id=crop_id, error=str(exc))
    try:
        return THUMBNAIL_CACHE.get_or_compute(
            resolved_image_path(image_path), tuple(bbox), size=VLM_CROP_SIZE
        )
    except Exception as exc:
        logger.warning('vlm_crop_thumbnail_failed', crop_id=crop_id, error=str(exc))
        return None


def load_region_jpeg(image_path: str, bbox: Sequence[float]) -> bytes:
    """One region's close-up (source-frame ``bbox``) at VLM size, as the
    region verifier sends it."""
    from src.services.curation.image_serving import THUMBNAIL_CACHE

    return THUMBNAIL_CACHE.get_or_compute(
        resolved_image_path(image_path), tuple(bbox), size=VLM_CROP_SIZE
    )


__all__ = [
    'CROP_JPEG_QUALITY',
    'VLM_CROP_SIZE',
    'cache_stats',
    'crop_jpeg_from_cache',
    'crop_jpeg_from_disk',
    'load_item_crop_jpeg',
    'load_region_jpeg',
    'load_vlm_item_jpeg',
    'resolved_image_path',
]

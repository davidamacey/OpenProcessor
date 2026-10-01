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

from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import get_curation_config
from src.core.logging import get_logger


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


__all__ = [
    'CROP_JPEG_QUALITY',
    'cache_stats',
    'crop_jpeg_from_cache',
    'crop_jpeg_from_disk',
    'load_item_crop_jpeg',
]

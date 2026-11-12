"""The padded context preview of an item's box: ``GET /crops/{crop_id}/preview``.

A thumbnail shows the box alone and never enlarges it, so the smallest boxes come back a few
pixels wide. The preview cuts the box with a margin of its own surroundings (``pad`` is a fraction
of the box's width and height added on every side, clamped to the image), scales the cut so its
longest side lands in ``[min_side, max_side]`` (up or down) and optionally outlines the box inside
it. Resampling is ``Image.Resampling.LANCZOS``: Pillow's handbook (Concepts, Filters) rates it high
quality for both upscaling and downscaling, the right filter for enlarging a small crop.

Cached like the thumbnails (:class:`~src.services.curation.image_serving.ThumbnailCache`): a
per-process LRU of the encoded JPEG, bounded in bytes, keyed by the image, the exact box and every
parameter, so an edited box can never be served from an old entry.
"""

from __future__ import annotations

import io
from threading import Lock
from typing import TYPE_CHECKING

from cachetools import LRUCache
from PIL import Image, ImageDraw, ImageOps

from src.utils.cpu_pool import run_cpu


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


PREVIEW_CACHE_MAX_BYTES = 16 * 1024 * 1024
"""Encoded bytes one process keeps (the stack runs many workers). Bounded by size, not entry
count: a preview may be up to 2048 px, so a count would not bound memory."""

OUTLINE_RGB = (255, 214, 10)
"""The highlight yellow (``--hl``): the colour the UI reserves for the selected thing."""
OUTLINE_WIDTH = 3

type _Key = tuple[str, tuple[float, float, float, float], float, int, int, bool]


def _normalized(bbox_norm: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox_norm) != 4:
        raise ValueError(f'bbox_norm must have 4 elements, got {len(bbox_norm)}')
    x1, y1, x2, y2 = (round(float(v), 6) for v in bbox_norm)
    return x1, y1, x2, y2


def _pixel_edge(low: float, high: float, size: int) -> tuple[int, int]:
    start, end = max(0, int(low * size)), min(size, int(high * size))
    if end <= start < size and high > low:
        end = start + 1  # a sub-pixel box still covers the pixel it sits in
    return start, end


def pixel_box(bbox_norm: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """The box in pixels, clamped to the image and at least one pixel on each side when it has
    any normalized area. ``ValueError`` for a box with no area or one outside the image."""
    x1, y1, x2, y2 = _normalized(bbox_norm)
    (bx1, bx2), (by1, by2) = _pixel_edge(x1, x2, width), _pixel_edge(y1, y2, height)
    if bx2 <= bx1 or by2 <= by1:
        raise ValueError(f'invalid bbox after pixel mapping: ({bx1},{by1},{bx2},{by2})')
    return bx1, by1, bx2, by2


def padded_pixel_box(
    bbox_norm: Sequence[float], width: int, height: int, pad: float
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """``(context, box)`` in pixels: the box and the box grown by ``pad`` of its size on every
    side, clamped to the image. ``ValueError`` for a box with no area."""
    bx1, by1, bx2, by2 = pixel_box(bbox_norm, width, height)
    mx, my = round((bx2 - bx1) * pad), round((by2 - by1) * pad)
    context = (max(0, bx1 - mx), max(0, by1 - my), min(width, bx2 + mx), min(height, by2 + my))
    return context, (bx1, by1, bx2, by2)


def target_scale(longest: int, min_side: int, max_side: int) -> float:
    """The factor that brings the context's longest side into ``[min_side, max_side]``: up to
    ``min_side`` when smaller, down to ``max_side`` when larger, else 1."""
    if longest < min_side:
        return min_side / longest
    if longest > max_side:
        return max_side / longest
    return 1.0


def render_preview(
    image_path: Path,
    bbox_norm: Sequence[float],
    *,
    pad: float,
    min_side: int,
    max_side: int,
    outline: bool,
) -> bytes:
    """Open, exif-rotate, cut the padded context, scale it with LANCZOS, outline the box when
    asked, JPEG-encode."""
    with Image.open(image_path) as src:
        img = ImageOps.exif_transpose(src)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        context, box = padded_pixel_box(bbox_norm, *img.size, pad)
        cut = img.crop(context)
    cw, ch = cut.size
    scale = target_scale(max(cw, ch), min_side, max_side)
    size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
    if size != cut.size:
        cut = cut.resize(size, Image.Resampling.LANCZOS)
    if outline:
        left, top = box[0] - context[0], box[1] - context[1]
        right, bottom = box[2] - context[0], box[3] - context[1]
        ImageDraw.Draw(cut).rectangle(
            (left * scale, top * scale, right * scale - 1, bottom * scale - 1),
            outline=OUTLINE_RGB,
            width=OUTLINE_WIDTH,
        )
    buf = io.BytesIO()
    cut.save(buf, format='JPEG', quality=88)
    return buf.getvalue()


class PreviewCache:
    """LRU cache of encoded previews (a per-process cache: concurrent misses for one key may each
    compute it once, the last write wins)."""

    def __init__(self, max_bytes: int = PREVIEW_CACHE_MAX_BYTES) -> None:
        # cachetools sizes each entry with ``getsizeof`` and evicts the least recently used until
        # the total fits ``maxsize``.
        self._cache: LRUCache[_Key, bytes] = LRUCache(maxsize=max_bytes, getsizeof=len)
        self._lock = Lock()

    @property
    def size_bytes(self) -> int:
        with self._lock:
            return int(self._cache.currsize)

    async def get_or_compute(
        self,
        image_path: Path,
        bbox_norm: Sequence[float],
        *,
        pad: float,
        min_side: int,
        max_side: int,
        outline: bool,
    ) -> bytes:
        key: _Key = (
            str(image_path),
            _normalized(bbox_norm),
            round(pad, 4),
            min_side,
            max_side,
            outline,
        )
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        jpeg = await run_cpu(
            render_preview,
            image_path,
            bbox_norm,
            pad=pad,
            min_side=min_side,
            max_side=max_side,
            outline=outline,
        )
        with self._lock:
            # cachetools refuses an entry larger than the whole cache; serve it uncached.
            if len(jpeg) <= self._cache.maxsize:
                self._cache[key] = jpeg
        return jpeg

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


PREVIEW_CACHE = PreviewCache()

__all__ = [
    'OUTLINE_RGB',
    'PREVIEW_CACHE',
    'PreviewCache',
    'padded_pixel_box',
    'pixel_box',
    'render_preview',
    'target_scale',
]

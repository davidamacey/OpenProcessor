"""Shared crop/bbox geometry helpers for the detection subsystem.

Extracted from the curation ingest service and from this repo's own
:mod:`src.services.detection.cascade_detect`, which previously each
carried their own near-identical copy of the letterbox/undo-letterbox
math. Both the curation ingest service
(:mod:`src.services.curation.ingest`) and the region-detection cascade
import these functions so bbox math, crop ids and letterbox transforms
stay bit-identical across the two pipelines.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np

from src.domain.image_geometry import letterbox_geometry
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from PIL import Image


def bbox_norm(
    bbox_pixel: tuple[float, float, float, float],
    width: int,
    height: int,
) -> list[float]:
    """Convert a pixel-space ``(x1, y1, x2, y2)`` bbox to normalized ``[0, 1]``.

    Clamped to the image bounds so a detector's slightly-out-of-frame box
    never produces an out-of-range stored value.
    """
    x1, y1, x2, y2 = bbox_pixel
    return [
        max(0.0, min(1.0, x1 / max(width, 1))),
        max(0.0, min(1.0, y1 / max(height, 1))),
        max(0.0, min(1.0, x2 / max(width, 1))),
        max(0.0, min(1.0, y2 / max(height, 1))),
    ]


def stored_bbox_norm(
    bbox_norm_values: tuple[float, float, float, float] | list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """The normalized box the index path stores for ``bbox_norm_values``.

    Indexing takes a pixel box and divides it back by the frame size, which
    can move a value by one float ulp; :func:`crop_id` rounds to 6 decimals,
    so a box on a rounding edge gets a different id. A caller that must
    predict the id an item will be stored under derives it from this.
    """
    x1, y1, x2, y2 = bbox_norm_values
    out = bbox_norm((x1 * width, y1 * height, x2 * width, y2 * height), width, height)
    return (out[0], out[1], out[2], out[3])


def crop_id(image_id: str, bbox_norm_values: list[float] | tuple[float, ...]) -> str:
    """Stable crop id = ``sha256(image_id + bbox)[:32]``.

    Deterministic in both the bbox values (6-decimal rounding) and their
    order, so the same ``(image_id, bbox)`` pair always yields the same
    id regardless of caller (ingest, label import, a backfill script).
    """
    x1, y1, x2, y2 = bbox_norm_values
    payload = f'{image_id}|{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}'
    return sha256(payload.encode('utf-8')).hexdigest()[:32]


def box_area(box: tuple[float, float, float, float]) -> float:
    """Area of an ``(x1, y1, x2, y2)`` box; an inverted side counts as zero."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Standard axis-aligned IoU on ``(x1, y1, x2, y2)`` boxes (any shared scale)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def letterbox_params(
    img: Image.Image,
    target: int,
) -> tuple[float, tuple[float, float]]:
    """The ``(scale, (pad_w, pad_h))`` :func:`letterbox_to_square` would produce.

    Pure arithmetic — no resize, no allocation. Callers that already hold
    a detector's raw output and only need the parameters to map boxes
    back to source coordinates use this instead of re-letterboxing the
    image just to throw the pixels away.

    Raises:
        ValueError: If ``img`` has a zero-size dimension.
    """
    lb = letterbox_geometry(*img.size, target)
    return lb.scale, (lb.pad_w, lb.pad_h)


def letterbox_to_square(
    img: Image.Image,
    target: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox (aspect-preserving pad) a PIL image to ``target`` x ``target``.

    Returns:
        Tuple ``(chw, scale, (pad_w, pad_h))``:

        * ``chw``: ``(1, 3, target, target)`` uint8 array of RGB pixels
          (the engine scales them to ``[0, 1]`` itself), ready for
          :func:`src.clients.triton_pool_core.images_input`.
        * ``scale``: Uniform scale factor applied to width and height.
        * ``(pad_w, pad_h)``: Pixel padding on the left/top edges (the
          canvas is centered, so right/bottom padding is implied).

    Raises:
        ValueError: If ``img`` has a zero-size dimension.
    """
    from PIL import Image as PILImage

    lb = letterbox_geometry(*img.size, target)

    with stage_timer('letterbox'):
        resized = img.resize((lb.new_w, lb.new_h), PILImage.Resampling.BILINEAR)

        canvas = PILImage.new('RGB', (target, target), fill)
        canvas.paste(resized, (lb.left, lb.top))

        chw = np.ascontiguousarray(np.asarray(canvas).transpose(2, 0, 1))[None]
        return chw, lb.scale, (lb.pad_w, lb.pad_h)


def crop_to_jpeg(crop: Image.Image, quality: int = 90) -> bytes:
    """Encode a PIL crop to JPEG bytes.

    Quality 90 balances small transfer size against preserving the
    high-frequency detail a downstream sub-region detector relies on.
    """
    import io

    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


__all__ = [
    'bbox_norm',
    'crop_id',
    'crop_to_jpeg',
    'iou',
    'letterbox_params',
    'letterbox_to_square',
]

"""Shared crop/bbox geometry helpers for the detection subsystem.

Extracted from the private reference ingest service (see
``docs/design/curation_design_rationale.md`` §2.1 for the citation
convention) and from this repo's own :mod:`src.services.detection.cascade_detect`,
which previously carried a private, near-identical copy of the
letterbox/undo-letterbox math. Both the curation ingest service
(:mod:`src.services.curation.ingest`) and the region-detection cascade
import these functions so bbox math, crop ids and letterbox transforms
stay bit-identical across the two pipelines.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np


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


def crop_id(image_id: str, bbox_norm_values: list[float] | tuple[float, ...]) -> str:
    """Stable crop id = ``sha256(image_id + bbox)[:32]``.

    Deterministic in both the bbox values (6-decimal rounding) and their
    order, so the same ``(image_id, bbox)`` pair always yields the same
    id regardless of caller (ingest, label import, a backfill script).
    """
    x1, y1, x2, y2 = bbox_norm_values
    payload = f'{image_id}|{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}'
    return sha256(payload.encode('utf-8')).hexdigest()[:32]


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
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
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
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        msg = f'degenerate image size: ({orig_w}, {orig_h})'
        raise ValueError(msg)
    scale = min(target / orig_h, target / orig_w)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    return float(scale), ((target - new_w) / 2.0, (target - new_h) / 2.0)


def letterbox_to_square(
    img: Image.Image,
    target: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox (aspect-preserving pad) a PIL image to ``target`` x ``target``.

    Returns:
        Tuple ``(chw, scale, (pad_w, pad_h))``:

        * ``chw``: ``(1, 3, target, target)`` FP32 array in ``[0, 1]``,
          ready for ``InferInput.set_data_from_numpy``.
        * ``scale``: Uniform scale factor applied to width and height.
        * ``(pad_w, pad_h)``: Pixel padding on the left/top edges (the
          canvas is centered, so right/bottom padding is implied).

    Raises:
        ValueError: If ``img`` has a zero-size dimension.
    """
    from PIL import Image as PILImage

    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        msg = f'degenerate image size: ({orig_w}, {orig_h})'
        raise ValueError(msg)

    scale = min(target / orig_h, target / orig_w)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = img.resize((new_w, new_h), PILImage.BILINEAR)

    canvas = PILImage.new('RGB', (target, target), fill)
    pad_w = (target - new_w) / 2.0
    pad_h = (target - new_h) / 2.0
    canvas.paste(resized, (int(pad_w), int(pad_h)))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return chw.astype(np.float32, copy=False), float(scale), (float(pad_w), float(pad_h))


def undo_letterbox(
    bbox_pixel: tuple[float, float, float, float],
    scale: float,
    pad: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Map a network-pixel-space bbox back to the pre-letterbox image's pixel coords."""
    pad_w, pad_h = pad
    x1, y1, x2, y2 = bbox_pixel
    s = max(scale, 1e-6)
    return (
        (x1 - pad_w) / s,
        (y1 - pad_h) / s,
        (x2 - pad_w) / s,
        (y2 - pad_h) / s,
    )


def crop_to_jpeg(crop: Image.Image, quality: int = 90) -> bytes:
    """Encode a PIL crop to JPEG bytes.

    Quality 90 balances small transfer size against preserving the
    high-frequency detail a downstream sub-region detector relies on.
    """
    import io

    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def roi_pool_sppf(
    sppf: np.ndarray,
    bbox_letterbox: tuple[float, float, float, float],
    *,
    input_size: int = 1280,
    target_dim: int = 1024,
) -> np.ndarray:
    """Average-pool a backbone feature map over a single detection bbox.

    ``sppf`` shape: ``(C, H, W)`` where ``H == W == input_size // stride``.
    ``bbox_letterbox`` is in letterboxed input coords (the same space the
    model saw). Projection: divide by stride, clamp to the grid, take the
    cells overlapping the bbox, mean across them -> ``C``-d vector, then
    zero-pad/truncate to ``target_dim`` and L2-normalize.

    Returns a ``(target_dim,)`` ``float32`` array. Safe for degenerate
    bboxes (falls back to the single grid cell containing the bbox center).
    """
    c, h, w = sppf.shape
    stride = input_size / float(h)
    x1, y1, x2, y2 = bbox_letterbox
    gx1 = max(0, min(w - 1, int(x1 / stride)))
    gy1 = max(0, min(h - 1, int(y1 / stride)))
    gx2 = max(gx1, min(w, int(np.ceil(x2 / stride))))
    gy2 = max(gy1, min(h, int(np.ceil(y2 / stride))))
    if gx2 <= gx1:
        gx2 = gx1 + 1
    if gy2 <= gy1:
        gy2 = gy1 + 1
    region = sppf[:, gy1:gy2, gx1:gx2]
    pooled = region.reshape(c, -1).mean(axis=1).astype(np.float32, copy=False)
    if c < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:c] = pooled
        pooled = padded
    elif c > target_dim:
        pooled = pooled[:target_dim]
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32, copy=False)


__all__ = [
    'bbox_norm',
    'crop_id',
    'crop_to_jpeg',
    'iou',
    'letterbox_params',
    'letterbox_to_square',
    'roi_pool_sppf',
    'undo_letterbox',
]

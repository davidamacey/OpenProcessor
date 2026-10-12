"""Canonical PE-Core-L14-336 image preprocessing for whole-frame embeddings.

Every code path that produces a *whole-frame* embedding (the images
index's global embedding field) should go through here, so ingest and
any backfill yield vectors in exactly the same space — a hard
requirement for cross-run near-duplicate detection and similar-image
kNN search.

Convention: resize so the shorter edge is 336, center-crop to
336x336, return CHW uint8. The scale to ``[0,1]`` and PE-Core's own mean/std
(0.5, 0.5 — upstream ``core.vision_encoder.transforms.get_image_transform``)
are applied inside the TensorRT engine (``export/uint8_input.py``), so the
request carries a quarter of the bytes of the float tensor.

Whole *frames* are decoded with OpenCV ``IMREAD_REDUCED_COLOR_8`` (a 1/8-scale
JPEG decode — fast and GIL-releasing, so a thread pool scales across cores) and
resized with ``cv2.INTER_LINEAR``.
"""

from __future__ import annotations

import cv2
import numpy as np


PE_SIZE = 336


def resize_crop_rgb(rgb: np.ndarray, target: int = PE_SIZE) -> np.ndarray:
    """Resize-shorter-edge + center-crop an HWC RGB array to ``target`` square."""
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target, target, 3), dtype=rgb.dtype)
    scale = target / min(h, w)
    nh, nw = max(target, round(h * scale)), max(target, round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (nh - target) // 2, (nw - target) // 2
    return resized[top : top + target, left : left + target]


def to_chw(rgb_uint8: np.ndarray) -> np.ndarray:
    """HWC-uint8 RGB -> contiguous CHW uint8 (a batch ``(N,H,W,3)`` -> ``(N,3,H,W)``).

    No arithmetic: the PE engine does ``(v / 255 - 0.5) / 0.5`` itself (see
    ``export/uint8_input.py``), so the request carries the pixels as decoded.
    """
    axes = (0, 3, 1, 2) if rgb_uint8.ndim == 4 else (2, 0, 1)
    return np.ascontiguousarray(rgb_uint8.transpose(axes))


def to_chw_into(out_chw: np.ndarray, rgb_uint8: np.ndarray) -> None:
    """:func:`to_chw` of one HWC image written into a contiguous ``(3,H,W)`` slot.

    Fills a preallocated batch in one pass instead of allocating, transposing and
    stacking one array per image.
    """
    out_chw[...] = rgb_uint8.transpose(2, 0, 1)


def whole_frame_rgb(path: str, target: int = PE_SIZE) -> np.ndarray | None:
    """cv2 1/8 decode + resize-shorter-edge + center-crop -> (target,target,3) RGB u8.

    The cheap, GIL-free part of whole-frame preprocessing — safe to call from
    a large read thread pool. Returns ``None`` on a decode failure.
    """
    bgr = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return resize_crop_rgb(rgb, target)


def whole_frame_rgb_from_bytes(data: bytes, target: int = PE_SIZE) -> np.ndarray | None:
    """:func:`whole_frame_rgb` for an in-memory encoded image (e.g. an upload).

    ``cv2.imdecode`` with the same ``IMREAD_REDUCED_COLOR_8`` flag runs the
    same decoder as ``cv2.imread`` on the same bytes, so an uploaded image
    and the identical file read from disk embed to the same vector.
    """
    if not data:
        return None
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return resize_crop_rgb(rgb, target)


def whole_frame_chw(path: str, target: int = PE_SIZE) -> np.ndarray | None:
    """Single-image convenience: decode + preprocess -> (3,target,target) uint8."""
    rgb = whole_frame_rgb(path, target)
    if rgb is None:
        return None
    return to_chw(rgb)


def whole_frame_chw_from_bytes(data: bytes, target: int = PE_SIZE) -> np.ndarray | None:
    """:func:`whole_frame_chw` for an in-memory encoded image."""
    rgb = whole_frame_rgb_from_bytes(data, target)
    if rgb is None:
        return None
    return to_chw(rgb)

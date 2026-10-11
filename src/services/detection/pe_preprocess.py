"""Canonical PE-Core-L14-336 image preprocessing for whole-frame embeddings.

Every code path that produces a *whole-frame* embedding (the images
index's global embedding field) should go through here, so ingest and
any backfill yield vectors in exactly the same space — a hard
requirement for cross-run near-duplicate detection and similar-image
kNN search.

Convention: resize so the shorter edge is 336, center-crop to
336x336, scale to ``[0,1]``, normalize with PE-Core's own mean/std (0.5, 0.5 —
upstream ``core.vision_encoder.transforms.get_image_transform``), return CHW
float32. The mean/std and the normalize math live here so a crop path
can share them too.

Whole *frames* are decoded with OpenCV ``IMREAD_REDUCED_COLOR_8`` (a 1/8-scale
JPEG decode — fast and GIL-releasing, so a thread pool scales across cores) and
resized with ``cv2.INTER_LINEAR``. The normalize is a single vectorized op so a
batch of frames can be normalized at once (keeps per-image numpy off the hot,
many-threaded read path).
"""

from __future__ import annotations

import cv2
import numpy as np


PE_SIZE = 336
# PE-Core was trained with Normalize([0.5]*3, [0.5]*3), not ImageNet statistics.
PE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
PE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)


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


def _build_normalize_lut() -> np.ndarray:
    """The normalized float32 of every uint8 value: ``(v / 255 - mean) / std``.

    The same three float32 operations the per-pixel arithmetic used to run on every
    frame, applied once to the 256 possible inputs, so a lookup returns exactly the
    value the arithmetic would have produced.
    """
    if len(set(PE_MEAN.ravel().tolist())) != 1 or len(set(PE_STD.ravel().tolist())) != 1:
        msg = 'a per-channel lookup table is needed when the channels differ'
        raise ValueError(msg)
    scaled = np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0
    return ((scaled - PE_MEAN[0, 0, 0]) / PE_STD[0, 0, 0]).astype(np.float32)


_NORMALIZE_LUT = _build_normalize_lut()


def normalize_chw(rgb_uint8: np.ndarray) -> np.ndarray:
    """PE-normalize HWC-uint8 RGB -> CHW float32.

    Accepts a single image ``(H,W,3)`` -> ``(3,H,W)`` or a batch
    ``(N,H,W,3)`` -> ``(N,3,H,W)``.
    """
    values = _NORMALIZE_LUT[rgb_uint8]
    return values.transpose(0, 3, 1, 2) if values.ndim == 4 else values.transpose(2, 0, 1)


def normalize_into(out_chw: np.ndarray, rgb_uint8: np.ndarray) -> None:
    """:func:`normalize_chw` of one HWC image written into a contiguous ``(3,H,W)`` slot.

    Fills a preallocated batch in one pass instead of allocating, transposing and
    stacking one array per image.
    """
    out_chw[...] = cv2.LUT(rgb_uint8, _NORMALIZE_LUT).transpose(2, 0, 1)


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
    """Single-image convenience: decode + preprocess -> (3,target,target) f32."""
    rgb = whole_frame_rgb(path, target)
    if rgb is None:
        return None
    return normalize_chw(rgb)


def whole_frame_chw_from_bytes(data: bytes, target: int = PE_SIZE) -> np.ndarray | None:
    """:func:`whole_frame_chw` for an in-memory encoded image."""
    rgb = whole_frame_rgb_from_bytes(data, target)
    if rgb is None:
        return None
    return normalize_chw(rgb)

"""Canonical PE-Core-L14-336 image preprocessing for whole-frame embeddings.

Every code path that produces a *whole-frame* embedding (the images
index's global embedding field) should go through here, so ingest and
any backfill yield vectors in exactly the same space — a hard
requirement for cross-run near-duplicate detection and similar-image
kNN search.

Convention: resize so the shorter edge is 336, center-crop to
336x336, scale to ``[0,1]``, ImageNet mean/std normalize, return CHW
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
PE_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
PE_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


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


def normalize_chw(rgb_uint8: np.ndarray) -> np.ndarray:
    """ImageNet-normalize HWC-uint8 RGB -> CHW float32.

    Accepts a single image ``(H,W,3)`` -> ``(3,H,W)`` or a batch
    ``(N,H,W,3)`` -> ``(N,3,H,W)``.
    """
    arr = rgb_uint8.astype(np.float32) / 255.0
    if arr.ndim == 4:
        chw = np.transpose(arr, (0, 3, 1, 2))
        return ((chw - PE_IMAGENET_MEAN[None]) / PE_IMAGENET_STD[None]).astype(
            np.float32, copy=False
        )
    chw = np.transpose(arr, (2, 0, 1))
    return ((chw - PE_IMAGENET_MEAN) / PE_IMAGENET_STD).astype(np.float32, copy=False)


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


def whole_frame_chw(path: str, target: int = PE_SIZE) -> np.ndarray | None:
    """Single-image convenience: decode + preprocess -> (3,target,target) f32."""
    rgb = whole_frame_rgb(path, target)
    if rgb is None:
        return None
    return normalize_chw(rgb)

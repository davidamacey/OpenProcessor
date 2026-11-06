"""CPU references of the shipped GPU image operations and the registered tensor gate.

The references are the product's own CPU twins: the letterbox and whole-frame square as Pillow
``BILINEAR`` (``src/services/detection/geometry.py``, ``src/utils/pe_preprocess.py``), the crop
squares as ``cv2.resize`` linear (``resize_crop_rgb``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageOps


if TYPE_CHECKING:
    from pathlib import Path


GATE_MEAN_ABS = 0.5
GATE_MAX_ABS = 8


def pillow_frame(path: Path) -> np.ndarray:
    """HWC uint8 RGB with EXIF orientation applied, as the product's CPU path decodes."""
    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img)
        return np.asarray(upright.convert('RGB'))


def letterbox_ref(frame: np.ndarray, net: int = 640, fill: int = 114) -> np.ndarray:
    """CHW uint8 letterbox canvas."""
    h, w = frame.shape[:2]
    scale = min(net / h, net / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray(frame).resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (net, net), (fill,) * 3)
    canvas.paste(resized, (int((net - nw) / 2.0), int((net - nh) / 2.0)))
    return np.ascontiguousarray(np.asarray(canvas).transpose(2, 0, 1))


def whole_square_ref(frame: np.ndarray, target: int = 336) -> np.ndarray:
    """CHW uint8 centre square after resizing the shorter edge to ``target`` (Pillow bilinear)."""
    h, w = frame.shape[:2]
    scale = target / min(h, w)
    nw, nh = max(target, round(w * scale)), max(target, round(h * scale))
    left, top = (nw - target) // 2, (nh - target) // 2
    sq = (
        Image.fromarray(frame)
        .resize((nw, nh), Image.Resampling.BILINEAR)
        .crop((left, top, left + target, top + target))
    )
    return np.ascontiguousarray(np.asarray(sq).transpose(2, 0, 1))


def crop_square_ref(
    frame: np.ndarray, rect: tuple[int, int, int, int], target: int = 336
) -> np.ndarray:
    """CHW uint8 square of one crop rectangle, plain bilinear (``resize_crop_rgb``)."""
    x1, y1, x2, y2 = rect
    crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
    h, w = crop.shape[:2]
    scale = target / min(h, w)
    nh, nw = max(target, round(h * scale)), max(target, round(w * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (nh - target) // 2, (nw - target) // 2
    return np.ascontiguousarray(
        resized[top : top + target, left : left + target].transpose(2, 0, 1)
    )


class Gate(NamedTuple):
    mean_abs: float
    max_abs: int
    passed: bool


def tensor_gate(ref: np.ndarray, cand: np.ndarray) -> Gate:
    """The registered tensor gate: mean abs diff < 0.5 grey and max <= 8 (09g section 6.4)."""
    if ref.shape != cand.shape:
        return Gate(float('inf'), 255, False)
    diff = np.abs(ref.astype(np.int16) - cand.astype(np.int16))
    mean, mx = float(diff.mean()), int(diff.max())
    return Gate(mean, mx, mean < GATE_MEAN_ABS and mx <= GATE_MAX_ABS)

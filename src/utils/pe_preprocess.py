"""Canonical PE-Core image preprocessing: whole frames and object crops.

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

A crop of an object goes to the crop encoder (:data:`~src.config.settings.TritonModelConfig.
PE_CROP_MODEL`, its own size): the WHOLE box, scaled to fit the square with its aspect kept (OpenCV
bilinear) and centred on a square of :data:`~src.domain.image_geometry.CROP_FILL`
(:func:`letterbox_crop_rgb`). The frame engine cuts the same square on the GPU
(``frame_engine/branches/embed.py``); both take the geometry from
:func:`~src.domain.image_geometry.letterbox_geometry`.

A whole frame is read from the full-resolution decoded frame (Pillow, EXIF orientation
applied): the shorter edge to 336 with Pillow ``BILINEAR`` (antialiased on a downscale), then the
centre square. Ingest passes the frame it already decoded for the detector, so a photo is decoded
once; the path and byte helpers decode the same way, so every caller lands in one space.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.domain.image_geometry import CROP_FILL, letterbox_geometry


PE_SIZE = 336


def letterbox_crop_rgb(rgb: np.ndarray, target: int, fill: int = CROP_FILL) -> np.ndarray:
    """The whole HWC RGB crop scaled to fit a ``target`` square (aspect kept), centred on ``fill``.

    A crop with no pixels is a square of ``fill``.
    """
    h, w = rgb.shape[:2]
    out = np.full((target, target, 3), fill, dtype=np.uint8)
    if h == 0 or w == 0:
        return out
    g = letterbox_geometry(w, h, target)
    out[g.top : g.top + g.new_h, g.left : g.left + g.new_w] = cv2.resize(
        rgb, (g.new_w, g.new_h), interpolation=cv2.INTER_LINEAR
    )
    return out


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


def whole_frame_rgb_from_image(img: Image.Image, target: int = PE_SIZE) -> np.ndarray:
    """Shorter edge to ``target`` (Pillow ``BILINEAR``) and the centre square of a decoded frame."""
    w, h = img.size
    if w == 0 or h == 0:
        return np.zeros((target, target, 3), dtype=np.uint8)
    rgb = img if img.mode == 'RGB' else img.convert('RGB')
    scale = target / min(w, h)
    nw, nh = max(target, round(w * scale)), max(target, round(h * scale))
    left, top = (nw - target) // 2, (nh - target) // 2
    square = rgb.resize((nw, nh), Image.Resampling.BILINEAR).crop(
        (left, top, left + target, top + target)
    )
    return np.asarray(square)


def _decode_upright(source: str | io.BytesIO) -> Image.Image | None:
    try:
        with Image.open(source) as img:
            return ImageOps.exif_transpose(img).convert('RGB')
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def whole_frame_rgb(path: str, target: int = PE_SIZE) -> np.ndarray | None:
    """Decode a file and return its whole-frame square; ``None`` on a decode failure."""
    img = _decode_upright(path)
    return None if img is None else whole_frame_rgb_from_image(img, target)


def whole_frame_rgb_from_bytes(data: bytes, target: int = PE_SIZE) -> np.ndarray | None:
    """:func:`whole_frame_rgb` for an in-memory encoded image."""
    if not data:
        return None
    img = _decode_upright(io.BytesIO(data))
    return None if img is None else whole_frame_rgb_from_image(img, target)


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

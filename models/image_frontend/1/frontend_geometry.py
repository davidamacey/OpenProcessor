"""Host-side logic of the GPU ingest front end: header routing, geometry and resampling weights.

Numpy only, so the offline test suite covers it (``tests/models/test_image_frontend_geometry.py``).
Every function reproduces the arithmetic of the CPU ingest path it stands in for:

* :func:`letterbox_geometry`  ``geometry.letterbox_to_square``
* :func:`whole_square_geometry`  ``pe_preprocess.whole_frame_rgb_from_image`` and
  ``pe_preprocess.resize_crop_rgb`` (shorter edge to 336, centre crop)
* :func:`crop_rect`  ``ingest_index.crop_pil`` (zeros outside the frame)
* :func:`blur_rect`  ``crop_quality.crop_lap_var``'s clamped rectangle
* :func:`filter_weights`  Pillow ``BILINEAR`` (antialiased: the support widens with the
  downscale factor) or, with ``antialias=False``, plain bilinear as OpenCV ``INTER_LINEAR``
"""

from __future__ import annotations

import struct
from typing import NamedTuple

import numpy as np


_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_PROGRESSIVE = {0xC2, 0xC6, 0xCA, 0xCE}
_LOSSLESS = {0xC3, 0xC7, 0xCB, 0xCF}
_STANDALONE = {0x01, *range(0xD0, 0xD8)}


class JpegHeader(NamedTuple):
    """What the decoder choice depends on, read from the markers without decoding."""

    kind: str
    width: int
    height: int
    components: int
    precision: int
    progressive: bool
    orientation: int
    lossless: bool


def _exif_orientation(payload: bytes) -> int:
    tiff = payload[6:]
    if len(tiff) < 8 or tiff[:2] not in (b'II', b'MM'):
        return 1
    end = '<' if tiff[:2] == b'II' else '>'
    (ifd,) = struct.unpack(end + 'I', tiff[4:8])
    if ifd + 2 > len(tiff):
        return 1
    (count,) = struct.unpack(end + 'H', tiff[ifd : ifd + 2])
    for k in range(count):
        entry = tiff[ifd + 2 + 12 * k : ifd + 14 + 12 * k]
        if len(entry) < 12:
            break
        tag, _typ, _n = struct.unpack(end + 'HHI', entry[:8])
        if tag == 0x0112:
            (value,) = struct.unpack(end + 'H', entry[8:10])
            return value if 1 <= value <= 8 else 1
    return 1


def jpeg_header(data: bytes) -> JpegHeader:
    """Parse ``data``'s markers up to the first frame header; ``kind`` is ``jpeg``, ``corrupt``
    (a JPEG start marker but no readable frame header) or ``other``."""
    if data[:2] != b'\xff\xd8':
        return JpegHeader('other', 0, 0, 0, 0, False, 1, False)
    orientation, i = 1, 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in _STANDALONE:
            i += 2
            continue
        (length,) = struct.unpack('>H', data[i + 2 : i + 4])
        payload = data[i + 4 : i + 2 + length]
        if marker == 0xE1 and payload[:6] == b'Exif\x00\x00':
            orientation = _exif_orientation(payload)
        elif marker in _SOF_MARKERS and len(payload) >= 6:
            precision, height, width, comps = struct.unpack('>BHHB', payload[:6])
            return JpegHeader(
                'jpeg', width, height, comps, precision, marker in _PROGRESSIVE, orientation,
                marker in _LOSSLESS,
            )  # fmt: skip
        elif marker == 0xDA:
            break
        i += 2 + length
    return JpegHeader('corrupt', 0, 0, 0, 0, False, orientation, False)


def route(data: bytes, min_megapixels: float, max_megapixels: float) -> tuple[str, JpegHeader]:
    """``('gpu', header)`` or ``(reason, header)``: the reason the image stays on the CPU path.

    Fail closed: the GPU decoders return an image for a truncated file, so a JPEG without its
    end-of-image marker is refused here and the CPU decoder rejects or accepts it as today.
    """
    h = jpeg_header(data)
    if h.kind != 'jpeg':
        return ('not_jpeg' if h.kind == 'other' else 'corrupt'), h
    if h.lossless or h.precision != 8:
        return 'precision', h
    if h.components not in (1, 3):
        return 'components', h
    if h.progressive:
        return 'progressive', h
    if not data.rstrip(b'\x00').endswith(b'\xff\xd9'):
        return 'truncated', h
    mp = h.width * h.height / 1e6
    if mp < min_megapixels:
        return 'small', h
    if mp > max_megapixels:
        return 'huge', h
    return 'gpu', h


def oriented_size(h: JpegHeader) -> tuple[int, int]:
    """``(width, height)`` after the EXIF orientation is applied."""
    return (h.height, h.width) if h.orientation in (5, 6, 7, 8) else (h.width, h.height)


class Letterbox(NamedTuple):
    new_w: int
    new_h: int
    left: int
    top: int
    scale: float
    pad_w: float
    pad_h: float


def letterbox_geometry(w: int, h: int, target: int) -> Letterbox:
    scale = min(target / h, target / w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    pad_w = (target - new_w) / 2.0
    pad_h = (target - new_h) / 2.0
    return Letterbox(new_w, new_h, int(pad_w), int(pad_h), float(scale), pad_w, pad_h)


class Square(NamedTuple):
    new_w: int
    new_h: int
    left: int
    top: int


def whole_square_geometry(w: int, h: int, target: int) -> Square:
    scale = target / min(h, w)
    nw, nh = max(target, round(w * scale)), max(target, round(h * scale))
    return Square(nw, nh, (nw - target) // 2, (nh - target) // 2)


def undo_letterbox(
    box: tuple[float, float, float, float], net: int, lb: Letterbox
) -> tuple[float, float, float, float]:
    """A detector box (normalized to the network input) in frame pixels, as
    ``WholeImageDetector.decode_primary_row`` maps it."""
    s = max(lb.scale, 1e-6)
    x1, y1, x2, y2 = (float(v) * net for v in box)
    return ((x1 - lb.pad_w) / s, (y1 - lb.pad_h) / s, (x2 - lb.pad_w) / s, (y2 - lb.pad_h) / s)


def crop_rect(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1i = max(0, round(x1))
    y1i = max(0, round(y1))
    return x1i, y1i, max(x1i + 1, round(x2)), max(y1i + 1, round(y2))


def blur_rect(
    bbox: tuple[float, float, float, float], w: int, h: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (round(v) for v in bbox)
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def filter_weights(
    in_size: int, out_size: int, *, antialias: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per output index: first source tap, tap count and the normalized weights
    ``[out_size, kmax]`` of a triangle filter centred on ``(i + 0.5) * in / out``."""
    scale = in_size / out_size
    support = max(scale, 1.0) if antialias else 1.0
    centers = (np.arange(out_size, dtype=np.float64) + 0.5) * scale
    first = np.maximum(np.floor(centers - support + 0.5), 0).astype(np.int64)
    last = np.minimum(np.floor(centers + support + 0.5), in_size).astype(np.int64)
    count = np.maximum(last - first, 0)
    kmax = max(1, int(count.max()))
    xs = first[:, None] + np.arange(kmax)[None, :]
    w = np.clip(1.0 - np.abs((xs - centers[:, None] + 0.5) / support), 0.0, None)
    w[np.arange(kmax)[None, :] >= count[:, None]] = 0.0
    total = w.sum(axis=1, keepdims=True)
    w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    return first.astype(np.int32), count.astype(np.int32), w.astype(np.float32)


def lap_var(n: int, total: int, squares: int) -> float | None:
    """``crop_quality.laplacian_var`` from the integer sums; ``None`` for an empty rectangle."""
    if n <= 0:
        return None
    return round((n * squares - total * total) / (n * n), 2)

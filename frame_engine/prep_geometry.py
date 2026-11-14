"""Host-side logic of the GPU decode: header routing, geometry and resampling weights.

Numpy only, so the offline test suite covers it (``tests/frame_engine/test_prep_geometry.py``).
Every function reproduces the arithmetic of the CPU ingest path it stands in for.

The letterbox, whole-square, un-map and crop-rectangle arithmetic is ``src/domain/image_geometry.py``,
the module the CPU paths use. Here:

* :func:`stored_box_rects`  the embed pass's ``crop_pil`` of a stored ``bbox_norm`` (#259)
* :func:`blur_rect`  ``crop_quality.crop_lap_var``'s clamped rectangle
* :func:`filter_weights`  Pillow ``BILINEAR`` (antialiased: the support widens with the
  downscale factor) or, with ``antialias=False``, plain bilinear as OpenCV ``INTER_LINEAR``
"""

from __future__ import annotations

import struct
from typing import NamedTuple

import numpy as np

from src.domain.image_geometry import crop_rect
from src.domain.jpeg_markers import (
    LOSSLESS,
    PROGRESSIVE,
    SOF_MARKERS,
    STANDALONE,
    exif_orientation,
    header_segments,
)


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
    end: int = 0  # bytes of the image proper, EOI included (set by route)


def jpeg_header(data: bytes) -> JpegHeader:
    """Parse ``data``'s markers up to the first frame header; ``kind`` is ``jpeg``, ``corrupt``
    (a JPEG start marker but no readable frame header) or ``other``."""
    if data[:2] != b'\xff\xd8':
        return JpegHeader('other', 0, 0, 0, 0, False, 1, False)
    orientation = 1
    for marker, payload in header_segments(data):
        if marker == 0xE1 and payload[:6] == b'Exif\x00\x00':
            orientation = exif_orientation(payload)
        elif marker in SOF_MARKERS and len(payload) >= 6:
            precision, height, width, comps = struct.unpack('>BHHB', payload[:6])
            return JpegHeader(
                'jpeg', width, height, comps, precision, marker in PROGRESSIVE, orientation,
                marker in LOSSLESS,
            )  # fmt: skip
    return JpegHeader('corrupt', 0, 0, 0, 0, False, orientation, False)


def _scan_markers(data: bytes) -> np.ndarray:
    """Offsets of the marker bytes that can end an entropy-coded scan, ascending.

    Inside a scan ``0xFF`` is stuffed (``FF00``), starts a restart marker (``FFD0``..``FFD7``) or
    is a fill byte before another ``0xFF``; an ``0xFF`` followed by any other byte is a marker
    that ends the scan. One vectorised pass (numpy releases the interpreter lock in it): the prep
    engine calls this from every worker thread, and a per-byte interpreter loop over a 7 MB photo
    held the lock for about 80 ms, which capped one process at about 7 photos per second.
    """
    a = np.frombuffer(data, np.uint8)
    ff = np.flatnonzero(a[:-1] == 0xFF)
    nxt = a[ff + 1]
    ends: np.ndarray = ff[(nxt != 0x00) & (nxt != 0xFF) & ((nxt < 0xD0) | (nxt > 0xD7))] + 1
    return ends


def jpeg_end(data: bytes) -> int | None:
    """Offset just past the end-of-image marker, or ``None`` when the file has none.

    Walks the marker segments (so an EOI inside a header segment, such as the EXIF thumbnail's,
    is skipped) and the entropy-coded scans, so bytes appended after the image (maker trailers, a
    motion photo's video) are not part of it.
    """
    n, i, in_scan = len(data), 2, False
    markers: np.ndarray | None = None
    while True:
        if in_scan:
            if markers is None:
                markers = _scan_markers(data)
            # The scan's first byte is at ``i``, so its ending marker byte is past ``i``.
            k = int(np.searchsorted(markers, i + 1))
            if k >= len(markers):
                return None
            i = int(markers[k])
            marker, i = data[i], i + 1
        else:
            if i >= n or data[i] != 0xFF:
                return None
            while i < n and data[i] == 0xFF:
                i += 1
            if i >= n:
                return None
            marker, i = data[i], i + 1
        if marker == 0xD9:
            return i
        if marker in STANDALONE:
            continue
        if i + 2 > n:
            return None
        (length,) = struct.unpack('>H', data[i : i + 2])
        i += length
        in_scan = marker == 0xDA


def route(data: bytes, min_megapixels: float, max_megapixels: float) -> tuple[str, JpegHeader]:
    """``('gpu', header)`` or ``(reason, header)``: the reason the image stays on the CPU path.

    Fail closed: the GPU decoders return an image for a truncated file, so a JPEG without its
    end-of-image marker is refused here and the CPU decoder rejects or accepts it as today. A
    complete JPEG with bytes after the marker goes to the GPU; ``header.end`` is where the
    decoder's input stops.
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
    end = jpeg_end(data)
    if end is None:
        return 'truncated', h
    h = h._replace(end=end)
    mp = h.width * h.height / 1e6
    if mp < min_megapixels:
        return 'small', h
    if mp > max_megapixels:
        return 'huge', h
    return 'gpu', h


def oriented_size(h: JpegHeader) -> tuple[int, int]:
    """``(width, height)`` after the EXIF orientation is applied."""
    return (h.height, h.width) if h.orientation in (5, 6, 7, 8) else (h.width, h.height)


def stored_box_rects(boxes_norm: np.ndarray, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """The rectangle of each stored ``bbox_norm`` (rows ``x1, y1, x2, y2`` in ``[0, 1]`` of the
    oriented frame): scaled to pixels in float64 as the CPU embed pass does, then :func:`crop_rect`."""
    rows = np.asarray(boxes_norm, np.float64).reshape(-1, 4)
    return [
        crop_rect((float(x1) * w, float(y1) * h, float(x2) * w, float(y2) * h), w, h)
        for x1, y1, x2, y2 in rows
    ]


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
    in_size: int, out_size: int, *, antialias: bool = True, lanczos: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per output index: first source tap, tap count and the normalized weights
    ``[out_size, kmax]`` of a triangle filter (``lanczos``: Pillow's ``LANCZOS``, ``a = 3``)
    centred on ``(i + 0.5) * in / out``."""
    scale = in_size / out_size
    stretch = max(scale, 1.0) if antialias else 1.0
    radius = 3.0 if lanczos else 1.0
    support = radius * stretch
    centers = (np.arange(out_size, dtype=np.float64) + 0.5) * scale
    first = np.maximum(np.floor(centers - support + 0.5), 0).astype(np.int64)
    last = np.minimum(np.floor(centers + support + 0.5), in_size).astype(np.int64)
    count = np.maximum(last - first, 0)
    kmax = max(1, int(count.max()))
    xs = first[:, None] + np.arange(kmax)[None, :]
    u = (xs - centers[:, None] + 0.5) / stretch
    if lanczos:
        w = np.where(np.abs(u) < radius, np.sinc(u) * np.sinc(u / radius), 0.0)
    else:
        w = np.clip(1.0 - np.abs(u), 0.0, None)
    w[np.arange(kmax)[None, :] >= count[:, None]] = 0.0
    total = w.sum(axis=1, keepdims=True)
    w = np.divide(w, total, out=np.zeros_like(w), where=total != 0)
    return first.astype(np.int32), count.astype(np.int32), w.astype(np.float32)


def lap_var(n: int, total: int, squares: int) -> float | None:
    """``crop_quality.laplacian_var`` from the integer sums; ``None`` for an empty rectangle."""
    if n <= 0:
        return None
    return round((n * squares - total * total) / (n * n), 2)

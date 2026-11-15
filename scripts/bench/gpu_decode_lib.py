"""Pure logic for the GPU decode probes (#217): JPEG header routing, the letterbox, PE and crop
geometry the GPU path must reproduce exactly, the tensor-level gate (T of the parent plan's
section 7), the CPU/GPU routing crossover and the end-to-end ceiling arithmetic.

No GPU, no network, no filesystem: the probes that touch the machine are in
``gpu_decode_bench.py`` and ``gpu_decode_parity.py``; every number they report goes through a
function here that ``tests/bench/test_gpu_decode_lib.py`` pins down.
"""

from __future__ import annotations

import math
import statistics
import struct
from typing import Any, NamedTuple

import numpy as np

from src.domain.jpeg_markers import (
    LOSSLESS,
    PROGRESSIVE,
    SOF_MARKERS,
    exif_orientation,
    header_segments,
)


# ------------------------------------------------------------------------------ JPEG header


class JpegHeader(NamedTuple):
    """What the decoder choice depends on, read from the header without decoding.

    ``kind`` is ``jpeg``, ``png``, ``webp``, ``heif``, ``gif``, ``corrupt`` (a JPEG start
    marker but no frame header) or ``unknown``.
    """

    kind: str
    width: int
    height: int
    components: int
    precision: int
    progressive: bool
    orientation: int
    has_icc: bool
    lossless: bool = False


def _non_jpeg_kind(data: bytes) -> str:
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    if data[4:8] == b'ftyp' and data[8:12] in (b'heic', b'heix', b'mif1', b'msf1', b'avif'):
        return 'heif'
    if data[:4] == b'GIF8':
        return 'gif'
    return 'unknown'


def jpeg_header(data: bytes) -> JpegHeader:
    """Parse the markers of ``data`` up to the first frame header (and its APP segments)."""
    if data[:2] != b'\xff\xd8':
        return JpegHeader(_non_jpeg_kind(data), 0, 0, 0, 0, False, 1, False)
    orientation, has_icc = 1, False
    for marker, payload in header_segments(data):
        if marker == 0xE1 and payload[:6] == b'Exif\x00\x00':
            orientation = exif_orientation(payload)
        elif marker == 0xE2 and payload[:12] == b'ICC_PROFILE\x00':
            has_icc = True
        elif marker in SOF_MARKERS and len(payload) >= 6:
            precision, height, width, comps = struct.unpack('>BHHB', payload[:6])
            return JpegHeader(
                'jpeg',
                width,
                height,
                comps,
                precision,
                marker in PROGRESSIVE,
                orientation,
                has_icc,
                marker in LOSSLESS,
            )
    return JpegHeader('corrupt', 0, 0, 0, 0, False, orientation, has_icc)


class RoutePolicy(NamedTuple):
    """Which images go to the GPU decoder; everything else is decoded on the CPU, counted."""

    min_megapixels: float
    max_megapixels: float
    progressive_on_gpu: bool


def route(header: JpegHeader, policy: RoutePolicy) -> str:
    """``gpu``, or ``cpu:<reason>`` with the reason the CPU fallback counter is keyed by."""
    if header.kind == 'corrupt':
        return 'cpu:corrupt'
    if header.kind != 'jpeg':
        return 'cpu:not_jpeg'
    if header.lossless or header.precision != 8:
        return 'cpu:precision'
    if header.components == 4:
        return 'cpu:cmyk'
    if header.progressive and not policy.progressive_on_gpu:
        return 'cpu:progressive'
    mp = header.width * header.height / 1e6
    if mp < policy.min_megapixels:
        return 'cpu:small'
    if mp > policy.max_megapixels:
        return 'cpu:huge'
    return 'gpu'


# ------------------------------------------------------------------------------- geometry


class PeSquare(NamedTuple):
    new_w: int
    new_h: int
    top: int
    left: int


def pe_square_geometry(w: int, h: int, target: int = 336) -> PeSquare:
    """The spike's square: resize the shorter edge to ``target``, centre crop (the frame square of
    ``image_geometry.whole_square_geometry``; the product no longer cuts crops this way)."""
    scale = target / min(h, w)
    nh, nw = max(target, round(h * scale)), max(target, round(w * scale))
    return PeSquare(nw, nh, (nh - target) // 2, (nw - target) // 2)


def centre_square_reference(rgb: np.ndarray, target: int = 336) -> np.ndarray:
    """The CPU reference of the spike's ``pe_square_u8``: OpenCV ``INTER_LINEAR`` on
    :func:`pe_square_geometry`. Bench-only: no product path cuts a crop this way."""
    import cv2

    g = pe_square_geometry(rgb.shape[1], rgb.shape[0], target)
    resized = cv2.resize(rgb, (g.new_w, g.new_h), interpolation=cv2.INTER_LINEAR)
    return resized[g.top : g.top + target, g.left : g.left + target]


# ---------------------------------------------------------------------------- tensor gate


class TensorDiff(NamedTuple):
    best_shift: tuple[int, int]
    mean_abs: float
    max_abs: int
    share_diff: float


def _overlap(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = a.shape[:2]
    ya, yb = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xa, xb = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    return a[ya, xa].astype(np.int16), b[yb, xb].astype(np.int16)


def tensor_diff(ref: np.ndarray, cand: np.ndarray, max_shift: int = 1) -> TensorDiff:
    """Compare two HWC uint8 tensors; the shift (rows, cols) of ``cand`` that best matches.

    The shift search catches an off-by-one pad or crop offset, which a mean difference alone
    would report as blur.
    """
    if ref.shape != cand.shape:
        msg = f'shape mismatch: {ref.shape} vs {cand.shape}'
        raise ValueError(msg)
    best: tuple[float, tuple[int, int]] | None = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            a, b = _overlap(cand, ref, dy, dx)
            score = float(np.abs(a - b).mean())
            if best is None or score < best[0] - 1e-12:
                best = (score, (dy, dx))
    assert best is not None
    full = np.abs(ref.astype(np.int16) - cand.astype(np.int16))
    return TensorDiff(best[1], float(full.mean()), int(full.max()), float((full > 0).mean()))


# --------------------------------------------------------------------- stats and ceilings


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {'n': 0}
    v = sorted(values)

    def pct(p: float) -> float:
        return v[min(len(v) - 1, max(0, math.ceil(p * len(v)) - 1))]

    return {
        'n': len(v),
        'median': statistics.median(v),
        'mean': statistics.fmean(v),
        'p05': pct(0.05),
        'p95': pct(0.95),
        'min': v[0],
        'max': v[-1],
    }


def _fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares ``ms = a + b * megapixels``."""
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    b, a = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def ceiling(
    *, host_cpu_s: float, host_cores: float, gpu_ms: list[tuple[float, int]]
) -> dict[str, Any]:
    """Images/s ceiling: the host (cores / CPU-s per image) or the GPU stages.

    ``gpu_ms`` is ``(ms per image, cards that share it)`` per GPU stage that runs serially on
    the same cards; the GPU ceiling is the slowest stage.
    """
    host = host_cores / host_cpu_s if host_cpu_s > 0 else math.inf
    gpu = min((n * 1000.0 / ms for ms, n in gpu_ms if ms > 0), default=math.inf)
    return {
        'host': host,
        'gpu': gpu,
        'images_per_s': min(host, gpu),
        'binding': 'host' if host < gpu else 'gpu',
    }

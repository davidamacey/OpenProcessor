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


# ------------------------------------------------------------------------------ JPEG header

_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_PROGRESSIVE = {0xC2, 0xC6, 0xCA, 0xCE}
_LOSSLESS = {0xC3, 0xC7, 0xCB, 0xCF}
_STANDALONE = {0x01, *range(0xD0, 0xD8)}


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


def _exif_orientation(payload: bytes) -> int:
    """Orientation tag (0x0112) of IFD0 of an APP1 ``Exif`` payload; 1 when absent."""
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
    orientation, has_icc, i = 1, False, 2
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
        elif marker == 0xE2 and payload[:12] == b'ICC_PROFILE\x00':
            has_icc = True
        elif marker in _SOF_MARKERS and len(payload) >= 6:
            precision, height, width, comps = struct.unpack('>BHHB', payload[:6])
            return JpegHeader(
                'jpeg',
                width,
                height,
                comps,
                precision,
                marker in _PROGRESSIVE,
                orientation,
                has_icc,
                marker in _LOSSLESS,
            )
        elif marker == 0xDA:
            break
        i += 2 + length
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


class Letterbox(NamedTuple):
    new_w: int
    new_h: int
    left: int
    top: int
    scale: float
    pad_w: float
    pad_h: float


def letterbox_geometry(w: int, h: int, target: int) -> Letterbox:
    """``letterbox_to_square``'s arithmetic: resize size, paste offset and the un-map pad."""
    scale = min(target / h, target / w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    pad_w = (target - new_w) / 2.0
    pad_h = (target - new_h) / 2.0
    return Letterbox(new_w, new_h, int(pad_w), int(pad_h), float(scale), pad_w, pad_h)


class PeSquare(NamedTuple):
    new_w: int
    new_h: int
    top: int
    left: int


def pe_square_geometry(w: int, h: int, target: int = 336) -> PeSquare:
    """``resize_crop_rgb``'s arithmetic: resize the shorter edge to ``target``, centre crop."""
    scale = target / min(h, w)
    nh, nw = max(target, round(h * scale)), max(target, round(w * scale))
    return PeSquare(nw, nh, (nh - target) // 2, (nw - target) // 2)


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

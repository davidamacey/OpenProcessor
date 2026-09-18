"""In-plane lean-angle recovery for a text-bearing region from OCR text-line geometry.

A region's own detector box (e.g. a license plate on a vehicle) is
axis-aligned, so it carries no rotation; the rotation lives in the
*text*. Given the rotated text-line quadrilaterals from a text
detector (e.g. PaddleOCR), this module measures the tilt of each
line's longest edge — the reading direction — relative to horizontal
and reports the length-weighted **median** across lines: the in-plane
lean a level camera would see. Because every line on a flat region
shares one tilt, the angle **spread** across lines doubles as a
frontal-vs-oblique confidence — lines agree on a straight-on region and
disagree once perspective shears it.

Pure ``cv2`` + ``numpy``; the caller supplies the detected polygons (this module
does not depend on any particular detector). Deskew by ``+lean`` to level a region.
"""

from __future__ import annotations

import functools
import math
from typing import NamedTuple

import cv2
import numpy as np


# A region read confidently below this line-angle spread (deg) is treated as a
# near-frontal view, where a single in-plane lean angle is well defined.
FRONTAL_MAX_SPREAD_DEG = 6.0


class TextLine(NamedTuple):
    """One detected text line: longest-edge tilt, that edge's length, the quad."""

    angle_deg: float
    length: float
    quad: np.ndarray


class LeanEstimate(NamedTuple):
    """Recovered region lean. ``angle_deg`` levels the region when deskewed."""

    angle_deg: float | None
    spread_deg: float
    text_lines: int
    frontal: bool


def longest_edge_angle(quad: np.ndarray) -> float:
    """Tilt (deg, in ``[-45, 45]``) of a text quad's longest edge; 0 = level.

    The longest edge of a text box runs along the reading direction. It is
    traversed left-to-right so the sign is stable: a positive angle means the
    line descends toward the right (image y points down).

    Args:
        quad: ``(N, 2)`` polygon vertices in pixel coordinates.

    Returns:
        The longest-edge tilt relative to horizontal, normalized to ``[-45, 45]``.
    """
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    edges = [(q[i], q[(i + 1) % len(q)]) for i in range(len(q))]
    a, b = max(edges, key=lambda e: float(np.hypot(*(e[1] - e[0]))))
    if b[0] < a[0]:
        a, b = b, a
    ang = math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0])))
    while ang <= -45:
        ang += 90
    while ang > 45:
        ang -= 90
    return ang


def lines_from_polys(polys) -> list[TextLine]:
    """Convert raw detector polygons into :class:`TextLine` records."""
    lines: list[TextLine] = []
    if polys is None:
        return lines
    for p in polys:
        q = np.asarray(p, dtype=np.float32).reshape(-1, 2)
        edges = [(q[i], q[(i + 1) % len(q)]) for i in range(len(q))]
        length = max(float(np.hypot(*(e[1] - e[0]))) for e in edges)
        lines.append(TextLine(longest_edge_angle(q), length, q))
    return lines


def lean_from_lines(
    lines: list[TextLine], *, keep_frac: float = 0.6
) -> tuple[float | None, float, int]:
    """Length-weighted median line angle, the angle spread, and lines kept.

    Keeps the longest ``keep_frac`` of lines (the main text, not tiny
    stickers/decals) and returns their median tilt. The spread (std) is a
    confidence: small => a flat, straight-on region whose lines agree; large =>
    perspective shear (an oblique view), where one lean angle is ill defined.

    Args:
        lines: detected text lines for one region.
        keep_frac: fraction of the longest lines to keep for the estimate.

    Returns:
        ``(median_angle_deg | None, spread_deg, n_lines_kept)``; the angle is
        ``None`` when no text lines were supplied.
    """
    if not lines:
        return None, 0.0, 0
    ordered = sorted(lines, key=lambda t: -t.length)
    keep = ordered[: max(1, round(len(ordered) * keep_frac))]
    angs = np.array([ln.angle_deg for ln in keep], dtype=float)
    return float(np.median(angs)), float(angs.std()), len(keep)


def estimate_from_lines(
    lines: list[TextLine], *, frontal_max_spread: float = FRONTAL_MAX_SPREAD_DEG
) -> LeanEstimate:
    """Build a :class:`LeanEstimate` from already-extracted text lines."""
    angle, spread, n = lean_from_lines(lines)
    frontal = angle is not None and spread <= frontal_max_spread
    return LeanEstimate(angle, spread, n, frontal)


def estimate_lean(polys, *, frontal_max_spread: float = FRONTAL_MAX_SPREAD_DEG) -> LeanEstimate:
    """High-level lean estimate from a region's detected text-line polygons.

    Args:
        polys: detected text-line quads (each ``(N, 2)``) from a text detector.
        frontal_max_spread: spread (deg) below which the region is treated as a
            near-frontal view with a well-defined single lean angle.

    Returns:
        A :class:`LeanEstimate`; ``angle_deg`` is ``None`` when no text was found.
    """
    return estimate_from_lines(lines_from_polys(polys), frontal_max_spread=frontal_max_spread)


def deskew(image: np.ndarray, angle_deg: float, *, border=(38, 38, 38)) -> np.ndarray:
    """Rotate ``image`` by ``angle_deg`` about its center to level the region.

    The lean from :func:`estimate_lean` is the rotation that flattens the
    region, so pass it directly.
    """
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)


# --- Detection-backed convenience ---------------------------------------------
# The functions above are pure cv2/numpy and take polygons the caller already
# has. The helpers below run the full pipeline -- text detection *and* angle
# recovery -- using PaddleOCR, imported lazily so importing this module stays
# cheap and dependency-free for callers that only need the geometry.


def text_polys(detector, image: np.ndarray) -> list:
    """Run a PaddleOCR-style text detector and return its line polygons."""
    out = detector.predict(image)
    polys = out[0].get('dt_polys') if out else None
    return [] if polys is None else list(polys)


@functools.cache
def default_detector():
    """Lazily build and cache a PaddleOCR ``TextDetection`` model (singleton)."""
    import os

    os.environ.setdefault('DISABLE_MODEL_SOURCE_CHECK', 'True')
    os.environ.setdefault('GLOG_minloglevel', '3')
    from paddleocr import TextDetection

    return TextDetection()


def recover_lean(
    image: np.ndarray, *, detector=None, frontal_max_spread: float = FRONTAL_MAX_SPREAD_DEG
) -> LeanEstimate:
    """Detect text and recover the region's in-plane lean in one call.

    Args:
        image: a BGR crop of the region (the detector reads small/distant
            regions better with some surrounding context).
        detector: a PaddleOCR-style detector exposing ``predict``; a cached
            default ``TextDetection`` is built lazily when omitted.
        frontal_max_spread: spread (deg) below which the region is treated as a
            near-frontal view with a well-defined single lean angle.

    Returns:
        A :class:`LeanEstimate`; ``angle_deg`` levels the region when deskewed.
    """
    det = detector or default_detector()
    return estimate_lean(text_polys(det, image), frontal_max_spread=frontal_max_spread)

"""Box normalization shared by the VOC and Open Images readers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


CLAMP_TOLERANCE_NORM = 1e-3
CLAMP_TOLERANCE_PX = 1.0


@dataclass(frozen=True)
class NormBox:
    box: tuple[float, float, float, float] | None
    """Normalized ``x1, y1, x2, y2``; ``None`` when ``problem`` is set."""
    problem: Literal['out_of_range', 'degenerate'] | None
    clamped: bool


def _clamp(v: float, limit: float, tolerance: float) -> tuple[float, bool] | None:
    if v < 0.0:
        return (0.0, True) if v >= -tolerance else None
    if v > limit:
        return (limit, True) if v <= limit + tolerance else None
    return v, False


def normalize_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    width: float = 1.0,
    height: float = 1.0,
    tolerance: float,
) -> NormBox:
    """Normalize a box given in the caller's units (pixels for VOC, already
    normalized for Open Images). A coordinate within ``tolerance`` outside the
    image is clamped; further out is ``out_of_range``. ``width`` and
    ``height`` must be positive."""
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return NormBox(None, 'out_of_range', False)
    clamped = False
    out: list[float] = []
    for v, limit in ((x1, width), (y1, height), (x2, width), (y2, height)):
        res = _clamp(v, limit, tolerance)
        if res is None:
            return NormBox(None, 'out_of_range', False)
        out.append(res[0])
        clamped = clamped or res[1]
    nx1, ny1, nx2, ny2 = out[0] / width, out[1] / height, out[2] / width, out[3] / height
    if nx2 <= nx1 or ny2 <= ny1:
        return NormBox(None, 'degenerate', clamped)
    return NormBox((nx1, ny1, nx2, ny2), None, clamped)


__all__ = ['CLAMP_TOLERANCE_NORM', 'CLAMP_TOLERANCE_PX', 'NormBox', 'normalize_box']

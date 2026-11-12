"""A box on the embedding scatter: ``x1,y1,x2,y2`` over the cached ``viz_x`` / ``viz_y`` coordinates."""

from __future__ import annotations

import math
from typing import Final


VIZ_BOX_MESSAGE: Final = 'viz_box needs x1,y1,x2,y2 with x1 <= x2 and y1 <= y2'

VizBox = tuple[float, float, float, float]


def check_viz_box(box: VizBox) -> VizBox:
    """``box`` when it is a finite, non-inverted rectangle; ``ValueError(VIZ_BOX_MESSAGE)`` otherwise."""
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) for v in box) or x1 > x2 or y1 > y2:
        raise ValueError(VIZ_BOX_MESSAGE)
    return box


def parse_viz_box(raw: str) -> VizBox:
    """``x1,y1,x2,y2``; ``ValueError(VIZ_BOX_MESSAGE)`` for anything else."""
    try:
        x1, y1, x2, y2 = (float(part) for part in raw.split(','))
    except ValueError as exc:
        raise ValueError(VIZ_BOX_MESSAGE) from exc
    return check_viz_box((x1, y1, x2, y2))

"""The size the detector reads: PaddleX ``DetResizeForTest.resize_image_type0`` (release/3.3).

``limit_type`` ``min`` raises the shorter side to ``side`` (never shrinks), ``max`` caps the longer
side at ``side`` (never enlarges); then a longer side past ``max_side`` is shrunk to it, and each
side is rounded to a multiple of 32 (at least 32).
"""

from __future__ import annotations


def det_size(h: int, w: int, side: int, limit: str, max_side: int) -> tuple[int, int]:
    """``(height, width)`` of the detector input for a frame ``h`` x ``w``."""
    if limit == 'max':
        ratio = float(side) / max(h, w) if max(h, w) > side else 1.0
    elif limit == 'min':
        ratio = (float(side) / min(h, w)) if min(h, w) < side else 1.0
    else:
        raise ValueError(f'unknown det_limit {limit!r}')
    rh, rw = int(h * ratio), int(w * ratio)
    if max(rh, rw) > max_side:
        shrink = float(max_side) / max(rh, rw)
        rh, rw = int(rh * shrink), int(rw * shrink)
    return max(int(round(rh / 32) * 32), 32), max(int(round(rw / 32) * 32), 32)

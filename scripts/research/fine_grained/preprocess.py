"""Square model inputs from a decoded crop, one function per registered preprocessing variant.

Variants (plan section 3): ``centre`` is the production rule (shorter edge to the target, centre
square), ``letterbox`` pads the longer edge to a square with mid-grey, ``margin10`` is ``centre``
on the box grown by 10 % per side (the box change is :func:`grow_box`), ``whole`` is ``centre`` on
the full frame. The resize filter is Pillow BILINEAR everywhere, as in ``src/utils/pe_preprocess.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image


if TYPE_CHECKING:
    from collections.abc import Sequence

VARIANTS = ('centre', 'letterbox', 'margin10', 'whole')
PAD_RGB = (128, 128, 128)
MARGIN = 0.10


def centre(img: Image.Image, target: int) -> np.ndarray:
    """Shorter edge to ``target`` and the centre square, as ``pe_preprocess`` does it."""
    w, h = img.size
    scale = target / min(w, h)
    nw, nh = max(target, round(w * scale)), max(target, round(h * scale))
    left, top = (nw - target) // 2, (nh - target) // 2
    sq = img.convert('RGB').resize((nw, nh), Image.Resampling.BILINEAR)
    return np.asarray(sq.crop((left, top, left + target, top + target)))


def letterbox(img: Image.Image, target: int) -> np.ndarray:
    """Longer edge to ``target``, pad the shorter symmetrically; nothing is cropped."""
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new('RGB', (target, target), PAD_RGB)
    canvas.paste(
        img.convert('RGB').resize((nw, nh), Image.Resampling.BILINEAR),
        ((target - nw) // 2, (target - nh) // 2),
    )
    return np.asarray(canvas)


def grow_box(
    box: Sequence[float], width: int, height: int, margin: float = MARGIN
) -> tuple[int, int, int, int]:
    """Pixel box ``(x1, y1, x2, y2)`` grown by ``margin`` of its size per side, clipped to the frame."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0, int(np.floor(x1 - mx))),
        max(0, int(np.floor(y1 - my))),
        min(width, int(np.ceil(x2 + mx))),
        min(height, int(np.ceil(y2 + my))),
    )


def variant_input(
    frame: Image.Image, box: Sequence[float] | None, variant: str, target: int
) -> np.ndarray:
    """HWC uint8 square for ``variant``; ``box`` is the pixel box of the labelled vehicle."""
    if variant == 'whole' or box is None:
        return centre(frame, target)
    w, h = frame.size
    if variant == 'margin10':
        return centre(frame.crop(grow_box(box, w, h)), target)
    crop = frame.crop(grow_box(box, w, h, 0.0))
    if variant == 'letterbox':
        return letterbox(crop, target)
    if variant == 'centre':
        return centre(crop, target)
    raise ValueError(f'unknown variant {variant!r}')

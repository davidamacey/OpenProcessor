"""Pixel geometry shared by the CPU paths and the frame engine (``frame_engine/``).

Pure arithmetic on sizes and boxes, standard library only: the frame engine's image carries
``src/domain`` and nothing else of ``src``.

Crops for the encoder are letterboxed (:data:`CROP_FILL`); frames keep the shorter-edge centre
square of :func:`whole_square_geometry`.

The resample of a crop onto its fitted rectangle (the one rule of the frame engine's kernel,
``antialias=True``, and of the API's host copy, Pillow ``BILINEAR``): a triangle filter whose
support widens with the downscale factor. A box larger than the encoder input is therefore
antialiased (area-weighted, every source pixel counts); a box smaller than the input gets the plain
bilinear enlargement, because the filter does not widen when it enlarges.

A letterbox pastes the resized frame at an integer offset, so the pad every caller un-maps with
is that same integer: ``(target - new) // 2``. An odd margin puts the extra pixel on the right
or bottom; a fractional pad would put every un-mapped coordinate half a pixel off.
"""

from __future__ import annotations

from typing import NamedTuple


#: The pad of an encoder crop square. PE's own transform has no pad (it squashes the image,
#: ``core.vision_encoder.transforms.get_image_transform``) and normalizes with mean 0.5 and std
#: 0.5, so the pad that stays at the normalized zero is 127.5; the integer 128.
CROP_FILL = 128


class Letterbox(NamedTuple):
    new_w: int
    new_h: int
    left: int
    top: int
    scale: float
    pad_w: float
    pad_h: float


def letterbox_geometry(w: int, h: int, target: int) -> Letterbox:
    """The aspect-preserving fit of a ``w`` x ``h`` frame into a ``target`` square.

    Raises:
        ValueError: if a side is not positive.
    """
    if w <= 0 or h <= 0:
        msg = f'degenerate image size: ({w}, {h})'
        raise ValueError(msg)
    scale = min(target / h, target / w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    left = (target - new_w) // 2
    top = (target - new_h) // 2
    return Letterbox(new_w, new_h, left, top, float(scale), float(left), float(top))


class Square(NamedTuple):
    new_w: int
    new_h: int
    left: int
    top: int


def whole_square_geometry(w: int, h: int, target: int) -> Square:
    """Shorter edge to ``target``, centre crop (the whole-frame embedding input only: a crop is
    letterboxed, :func:`letterbox_geometry`)."""
    scale = target / min(h, w)
    nw, nh = max(target, round(w * scale)), max(target, round(h * scale))
    return Square(nw, nh, (nw - target) // 2, (nh - target) // 2)


def undo_letterbox(
    bbox_pixel: tuple[float, float, float, float], scale: float, pad: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Map a network-pixel-space box back to the pre-letterbox frame's pixels."""
    pad_w, pad_h = pad
    x1, y1, x2, y2 = (float(v) for v in bbox_pixel)
    s = max(scale, 1e-6)
    return ((x1 - pad_w) / s, (y1 - pad_h) / s, (x2 - pad_w) / s, (y2 - pad_h) / s)


def crop_rect(
    bbox_pixel: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """The integer pixel rectangle ``(x1, y1, x2, y2)`` (x2/y2 exclusive) that holds a stored box:
    rounded, cut to the frame, at least one pixel wide and high."""
    x1, y1, x2, y2 = bbox_pixel
    x1i = min(max(0, round(x1)), max(width, 1) - 1)
    y1i = min(max(0, round(y1)), max(height, 1) - 1)
    return (
        x1i,
        y1i,
        min(max(x1i + 1, round(x2)), max(width, 1)),
        min(max(y1i + 1, round(y2)), max(height, 1)),
    )

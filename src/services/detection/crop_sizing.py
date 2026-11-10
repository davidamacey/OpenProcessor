"""The work crop's size: one function that caps an item crop's long side.

Pillow ``Image.thumbnail`` resizes in place, keeps the aspect ratio, only ever shrinks, and
reduces in two steps (``reducing_gap``, JPEG draft/box reduce then a convolution resample), so the
result is antialiased (Pillow reference: ``PIL.Image.Image.thumbnail``, ``Image.resize``,
https://pillow.readthedocs.io/en/stable/reference/Image.html). LANCZOS matches the thumbnail
route in ``image_serving``.

``max_side`` 0 means "no cap": the crop is returned untouched, so a deployment that leaves the
setting at its default gets today's bytes.
"""

from __future__ import annotations

from PIL import Image


def cap_crop(crop: Image.Image, max_side: int) -> Image.Image:
    """``crop`` with its long side at most ``max_side`` pixels (aspect kept, never upscaled).

    Returns ``crop`` itself when ``max_side`` is 0 or the crop already fits; otherwise a new
    image (the argument is not modified)."""
    if max_side <= 0 or max(crop.size) <= max_side:
        return crop
    out = crop.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out

"""Embedded ICC profiles to sRGB: the one rule both decode routes follow (littleCMS through Pillow).

The models are trained on sRGB pixels. A photo tagged with another RGB profile (Display P3 from a
phone, Adobe RGB from a camera) holds different numbers for the same colours, so its pixels are
converted to sRGB before any model or rendition reads them. littleCMS does the conversion
(``PIL.ImageCms``, Pillow's bundled lcms2; perceptual intent and no flags, Pillow's defaults,
https://pillow.readthedocs.io/en/stable/reference/ImageCms.html). The CPU route converts each
photo with :func:`to_srgb`; the frame engine applies the table of :func:`classify`, the same transform
evaluated at every 8-bit input, on the GPU, so both routes give the same pixels.

What a profile is (:func:`profile_kind`, remembered per process by the profile's bytes: the answer
is a pure function of them, so the memo is never stale):

* ``srgb``: littleCMS maps it to sRGB within one level at every 8-bit input (the common sRGB
  IEC61966-2.1 profiles differ from Pillow's built-in sRGB only by the rounding of their tone
  curves). Not converted: the pixels stay exactly as decoded, as they did before this rule.
* ``convert``: an RGB profile that moves some input by more than one level.
* ``other``: not an RGB profile (grey, CMYK) or not a profile littleCMS can open. Ignored, as
  before: CMYK photos keep their own CPU conversion and a broken profile does not refuse the photo.
"""

from __future__ import annotations

import functools
import io
from typing import Literal

import numpy as np
from PIL import Image, ImageCms


ProfileKind = Literal['srgb', 'convert', 'other']

#: Levels a profile may move any 8-bit input and still count as sRGB.
SRGB_TOLERANCE = 1
#: Profiles whose kind is remembered (one entry is a profile's bytes and a word).
KIND_CACHE = 256

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))


def _transform(icc: bytes) -> ImageCms.ImageCmsTransform | None:
    """The profile-to-sRGB transform, or ``None`` for a profile that is not RGB or does not open."""
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    except (OSError, ImageCms.PyCMSError, ValueError, TypeError):
        return None
    if prof.profile.xcolor_space != 'RGB ':
        return None
    try:
        return ImageCms.buildTransform(prof, _SRGB, 'RGB', 'RGB')
    except ImageCms.PyCMSError:
        return None


def _all_inputs() -> Image.Image:
    """Every 8-bit RGB value once, ``r`` major: the pixel at flat index ``r*65536 + g*256 + b``."""
    v = np.arange(1 << 24, dtype=np.uint32)
    rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb.reshape(4096, 4096, 3))


def classify(icc: bytes) -> tuple[ProfileKind, np.ndarray | None]:
    """The profile's kind and, for ``convert``, the conversion at every 8-bit input:
    ``uint8 [2**24, 3]`` indexed ``r*65536 + g*256 + b`` (littleCMS's answer for each input).
    Builds the table (about 2 s of one core); not remembered, so callers keep what they need."""
    transform = _transform(icc)
    if transform is None:
        return 'other', None
    inputs = _all_inputs()
    table = np.asarray(transform.apply(inputs)).reshape(1 << 24, 3)
    moved = np.abs(table.astype(np.int16) - np.asarray(inputs).reshape(1 << 24, 3)).max()
    if int(moved) <= SRGB_TOLERANCE:
        return 'srgb', None
    return 'convert', table


@functools.lru_cache(maxsize=KIND_CACHE)
def _kind(icc: bytes) -> ProfileKind:
    return classify(icc)[0]


def profile_kind(icc: bytes | None) -> ProfileKind:
    """What the decode routes do with a photo tagged ``icc`` (``None``: untagged, sRGB)."""
    return _kind(icc) if icc else 'srgb'


def to_srgb(img: Image.Image, icc: bytes | None) -> Image.Image:
    """``img`` (mode ``RGB``) in sRGB: converted for a ``convert`` profile, else ``img`` itself."""
    if img.mode != 'RGB' or not icc or profile_kind(icc) != 'convert':
        return img
    transform = _transform(icc)
    return img if transform is None else transform.apply(img)

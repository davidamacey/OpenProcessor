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

import io
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Literal

import numpy as np
from PIL import Image, ImageCms

from src.utils.icc_curves import tone_curves


ProfileKind = Literal['srgb', 'convert', 'other']

#: Levels a profile may move any 8-bit input and still count as sRGB.
SRGB_TOLERANCE = 1
#: Profiles whose kind is remembered (one entry is a profile's bytes and a word).
KIND_CACHE = 256
#: Largest difference of any of the nine matrix entries (PCS XYZ) from sRGB's that still lets the
#: tone curves decide, and the one past which the matrix alone decides ``convert`` (a primary 2 %
#: of the white off moves mixed colours by several levels whatever the curves do).
COLORANT_NEAR = 0.002
COLORANT_FAR = 0.02
#: With the matrix at sRGB's, the largest move the curves alone cause at any of the 256 input
#: levels: at or under the first the profile is ``srgb``, at or over the second it is ``convert``;
#: in between the transform decides (littleCMS's own 8-bit rounding is of that order).
CURVE_LEVELS_SRGB = 0.5
CURVE_LEVELS_CONVERT = 1.5

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))


def _colorants(prof: ImageCms.core.CmsProfile) -> np.ndarray | None:
    """The nine matrix entries (PCS XYZ of the red, green and blue colorants), or ``None`` when
    the profile is not a matrix/TRC one (a lookup-table profile has no such matrix to compare)."""
    try:
        if not prof.is_matrix_shaper:
            return None
        found = [prof.red_colorant, prof.green_colorant, prof.blue_colorant]
    except (ImageCms.PyCMSError, TypeError):
        return None
    xyz = [c[0] for c in found if c is not None]
    return np.array(xyz) if len(xyz) == 3 else None


_SRGB_COLORANTS = _colorants(_SRGB.profile)


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


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2.1 linear light to the sRGB-encoded value (the curve of littleCMS's built-in sRGB)."""
    v = np.clip(linear, 0.0, 1.0)
    out: np.ndarray = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return out


def _curve_move(icc: bytes) -> float | None:
    """With the matrix at sRGB's, the largest level (of 255) the profile's tone curves move any of
    the 256 input levels from the sRGB curve; ``None`` when the curves cannot be read."""
    curves = tone_curves(icc)
    if curves is None:
        return None
    x = np.arange(256) / 255.0
    moved = max(float(np.abs(255.0 * _srgb_encode(c(x)) - 255.0 * x).max()) for c in curves)
    return moved if np.isfinite(moved) else None


def tag_verdict(icc: bytes) -> ProfileKind | None:
    """The kind the profile's own tags settle, or ``None`` when only the transform can.

    Not RGB or not openable: ``other`` (the transform would not be built either). A matrix/TRC
    profile (three colorants and three tone curves, nothing else decides its colours) is compared
    with sRGB numerically: the matrix first (Pillow's ``red_colorant``, ``green_colorant``,
    ``blue_colorant``) and, when the matrix is sRGB's, the tone curves (``icc_curves``). A profile
    with lookup tables, or a matrix or curve in the undecided band between the constants above,
    gives ``None``. Opens the profile once and builds no transform.
    """
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc)).profile
    except (OSError, ImageCms.PyCMSError, ValueError, TypeError):
        return 'other'
    if prof.xcolor_space != 'RGB ':
        return 'other'
    colorants = _colorants(prof)
    if colorants is None or _SRGB_COLORANTS is None:
        return None
    distance = float(np.abs(colorants - _SRGB_COLORANTS).max())
    if distance > COLORANT_FAR:
        return 'convert'
    if distance > COLORANT_NEAR:
        return None
    moved = _curve_move(icc)
    if moved is None:
        return None
    if moved <= CURVE_LEVELS_SRGB:
        return 'srgb'
    return 'convert' if moved >= CURVE_LEVELS_CONVERT else None


def _all_inputs() -> Image.Image:
    """Every 8-bit RGB value once, ``r`` major: the pixel at flat index ``r*65536 + g*256 + b``."""
    v = np.arange(1 << 24, dtype=np.uint32)
    rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb.reshape(4096, 4096, 3))


def classify(icc: bytes) -> tuple[ProfileKind, np.ndarray | None]:
    """The profile's kind and, for ``convert``, the conversion at every 8-bit input:
    ``uint8 [2**24, 3]`` indexed ``r*65536 + g*256 + b`` (littleCMS's answer for each input).
    A kind the tags settle (:func:`tag_verdict`) costs no table unless it is ``convert``; the
    table (about 1 s of one core) is not remembered, so callers keep what they need."""
    return _classify(icc, tag_verdict(icc))


def _classify(icc: bytes, verdict: ProfileKind | None) -> tuple[ProfileKind, np.ndarray | None]:
    if verdict in ('srgb', 'other'):
        return verdict, None
    transform = _transform(icc)
    if transform is None:
        return 'other', None
    inputs = _all_inputs()
    table = np.asarray(transform.apply(inputs)).reshape(1 << 24, 3)
    if verdict == 'convert':
        return 'convert', table
    moved = np.abs(table.astype(np.int16) - np.asarray(inputs).reshape(1 << 24, 3)).max()
    if int(moved) <= SRGB_TOLERANCE:
        return 'srgb', None
    return 'convert', table


_kinds_lock = threading.Lock()
_kinds: OrderedDict[bytes, Future[ProfileKind]] = OrderedDict()


def _decide(icc: bytes) -> ProfileKind:
    verdict = tag_verdict(icc)
    return verdict if verdict is not None else _classify(icc, None)[0]


def _kind(icc: bytes) -> ProfileKind:
    """The kind of ``icc``, computed once per process: a thread that meets a profile another
    thread is still classifying waits for that answer (``functools.lru_cache`` would compute it
    again in every such thread: its documentation says the function may be called more than once
    when a second call comes before the first is cached). The memo is bounded by ``KIND_CACHE``
    (oldest answered entry out first); a computation that raises is not remembered."""
    with _kinds_lock:
        pending = _kinds.get(icc)
        mine = pending is None
        if pending is None:
            pending = _kinds[icc] = Future()
            for old in [k for k, f in _kinds.items() if f.done()][
                : max(0, len(_kinds) - KIND_CACHE)
            ]:
                del _kinds[old]
        else:
            _kinds.move_to_end(icc)
    if not mine:
        return pending.result()
    try:
        kind = _decide(icc)
    except BaseException as exc:
        with _kinds_lock:
            del _kinds[icc]
        pending.set_exception(exc)
        raise
    pending.set_result(kind)
    return kind


def profile_kind(icc: bytes | None) -> ProfileKind:
    """What the decode routes do with a photo tagged ``icc`` (``None``: untagged, sRGB)."""
    return _kind(icc) if icc else 'srgb'


def to_srgb(img: Image.Image, icc: bytes | None) -> Image.Image:
    """``img`` (mode ``RGB``) in sRGB: converted for a ``convert`` profile, else ``img`` itself."""
    if img.mode != 'RGB' or not icc or profile_kind(icc) != 'convert':
        return img
    transform = _transform(icc)
    return img if transform is None else transform.apply(img)

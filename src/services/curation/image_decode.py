"""Decode an uploaded or ingested image once, into the two forms ingest reads.

JPEG (the bulk of photo ingest) goes straight to libjpeg-turbo through ``simplejpeg``: the
same library and settings Pillow uses (accurate DCT, smooth chroma upsampling), so the pixels are
identical, at about half of Pillow's CPU on a 12-24 MP photo, and the result is a numpy array the
blur scoring reads without the 54 MB copy ``np.asarray(PIL image)`` makes. Pillow still holds the
frame for the letterbox, the whole-frame square and the crops (one copy into its own layout).
Everything else, CMYK JPEGs and a JPEG whose header libjpeg-turbo cannot read (no
subsampling level) included, decodes through Pillow.

Damaged files are refused as before: ``simplejpeg`` raises on libjpeg's recoverable errors
(``strict``), so a truncated JPEG fails the decode instead of ingesting a grey tail.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


EXIF_ORIENTATION = 0x0112

NOT_DECODABLE = 'not a decodable image'
HEADER_REJECTED = 'Could not determine subsampling level'


class ImageDecodeError(ValueError):
    """The bytes do not decode as an image; ``str(exc)`` is the stable reason and the
    original error stays chained as ``__cause__``."""


def _decode_reason(exc: BaseException) -> str:
    """Ultralytics replaces ``PIL.Image.open`` with a wrapper that, when Pillow fails,
    imports ``pi_heif`` and raises that import error instead (Pillow's own error is
    only its ``__context__``). A Pillow "cannot identify image file" anywhere in the
    chain is the real failure: report it as :data:`NOT_DECODABLE`, not the import error."""
    link: BaseException | None = exc
    while link is not None:
        if isinstance(link, UnidentifiedImageError):
            return NOT_DECODABLE
        link = link.__context__
    return str(exc) or type(exc).__name__


@dataclass(frozen=True)
class DecodedImage:
    """One decoded frame, EXIF orientation applied. ``rgb`` holds the same pixels as ``pil``
    (HWC uint8) when the decoder produced an array, else ``None``."""

    pil: Image.Image
    rgb: np.ndarray | None = None


def orient_array(rgb: np.ndarray, orientation: int) -> np.ndarray:
    """Re-orient an HWC array by its EXIF tag as ``ImageOps.exif_transpose`` does."""
    out = rgb
    if orientation in (5, 6, 7, 8):
        out = out.transpose(1, 0, 2)
    if orientation in (2, 3, 6, 7):
        out = out[:, ::-1]
    if orientation in (3, 4, 7, 8):
        out = out[::-1]
    return out if out is rgb else np.ascontiguousarray(out)


def decode_image(data: bytes) -> DecodedImage:
    """Decode ``data`` (any format Pillow reads) into an upright RGB frame; raises
    :class:`ImageDecodeError` on a file that does not decode."""
    try:
        return _decode(data)
    except Exception as exc:
        raise ImageDecodeError(_decode_reason(exc)) from exc


def _decode(data: bytes) -> DecodedImage:
    header = Image.open(io.BytesIO(data))
    orientation = header.getexif().get(EXIF_ORIENTATION, 1)
    if header.format == 'JPEG' and header.mode in ('RGB', 'L'):
        import simplejpeg

        try:
            decoded = simplejpeg.decode_jpeg(data, colorspace='RGB')
        except ValueError as exc:
            # libjpeg-turbo's header read refuses some JPEGs Pillow reads fine (it cannot
            # determine the subsampling). That, and only that, takes the Pillow path; a
            # damaged file (any other error) still fails.
            if HEADER_REJECTED not in str(exc):
                raise
        else:
            rgb = orient_array(decoded, orientation)
            return DecodedImage(Image.fromarray(rgb), rgb)
    img = ImageOps.exif_transpose(header) if orientation != 1 else header
    if img.mode != 'RGB':
        img = img.convert('RGB')
    else:
        img.load()
    return DecodedImage(img)

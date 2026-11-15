"""The decode of the images nvJPEG does not take: small, progressive, CMYK, PNG, WebP and the rest.

One function decodes them, the API's own: :func:`src.utils.image_decode.decode_image` (simplejpeg
for JPEG, Pillow for everything else, EXIF orientation applied, an embedded RGB profile converted
to sRGB by littleCMS, ``src/utils/icc_srgb.py``). The frame it returns is therefore the one the API's
CPU path decodes today, upright and already in sRGB: the engine uploads it as it is and runs no
orientation and no colour step on it (``engine.py``).

Size first. :func:`probe` reads the dimensions from the header (``Image.open`` is lazy: "the actual
image data is not read from the file until you try to process the data", Pillow's ``Image.open``
documentation, https://pillow.readthedocs.io/en/stable/reference/Image.html) so the engine refuses
an image by its pixel count before a byte of it is decoded. Past twice ``Image.MAX_IMAGE_PIXELS``
``Image.open`` itself raises ``DecompressionBombError`` ("If the number of pixels is greater than
twice MAX_IMAGE_PIXELS, then a DecompressionBombError will be raised", same page); that is
:class:`TooLargeError` here, the engine's ``huge`` answer.

Threads: the decode runs on an engine worker thread and does not hold the interpreter lock for its
native part. simplejpeg calls ``tjDecompress2`` inside ``with nogil:``
(https://github.com/jfolz/simplejpeg/blob/master/simplejpeg/_jpeg.pyx) and Pillow's decoder wrapper
brackets ``decoder->decode`` with ``ImagingSectionEnter``, which is ``PyEval_SaveThread``
(``src/decode.c`` and ``src/_imaging.c``), except for the decoders that pull from the file
(JPEG 2000, SGI RLE), which keep it. The Python between the native calls (the orientation copy,
``Image.fromarray``, ``np.asarray``) is numpy and Pillow copies that also release it.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np
from PIL import Image

from src.utils.image_decode import (
    EXIF_ORIENTATION,
    IMAGE_DECODE_ERRORS,
    ImageDecodeError,
    decode_image,
)


class HostDecodeError(ValueError):
    """The bytes are not an image any decoder here reads."""


class TooLargeError(HostDecodeError):
    """The header says more pixels than Pillow will open at all."""


class HostHeader(NamedTuple):
    """The size of the upright frame."""

    width: int
    height: int


def probe(data: bytes) -> HostHeader:
    """The upright frame's size, from the header only.

    ``Image.getexif()`` is not used: for a PNG it loads the whole image, because the format allows
    its ``eXIf`` chunk after the pixel data. The orientation is read from the chunk or segment the
    open already parsed; a PNG with a late ``eXIf`` reports its stored size, the pixel count (what
    the budget reserves) is the same either way and the engine takes the decoded shape as final.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            raw = img.info.get('exif')
            orientation = 1
            if raw:
                exif = Image.Exif()
                exif.load(raw)
                orientation = exif.get(EXIF_ORIENTATION, 1)
    except Image.DecompressionBombError as exc:
        raise TooLargeError(str(exc)) from exc
    except IMAGE_DECODE_ERRORS as exc:
        raise HostDecodeError(str(exc) or type(exc).__name__) from exc
    return HostHeader(height, width) if orientation in (5, 6, 7, 8) else HostHeader(width, height)


def decode(data: bytes) -> np.ndarray:
    """The upright sRGB frame, HWC uint8, C-contiguous."""
    try:
        decoded = decode_image(data)
    except ImageDecodeError as exc:
        raise HostDecodeError(str(exc)) from exc
    rgb = decoded.rgb if decoded.rgb is not None else np.asarray(decoded.pil)
    return np.ascontiguousarray(rgb)

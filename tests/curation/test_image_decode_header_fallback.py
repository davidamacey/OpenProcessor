"""A JPEG the fast decoder's header read rejects (``tjDecompressHeader3(): Could not determine
subsampling level``) still decodes through Pillow, which reads it; other fast-decoder errors
(a damaged file) keep failing."""

from __future__ import annotations

import io

import numpy as np
import pytest
import simplejpeg
from PIL import Image

from src.services.curation.image_decode import ImageDecodeError, decode_image


HEADER_ERROR = 'tjDecompressHeader3(): Could not determine subsampling level of JPEG image'


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((16, 24, 3), 90, dtype=np.uint8)).save(buf, 'JPEG')
    return buf.getvalue()


def _reject(message: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_a: object, **_k: object) -> None:
        raise ValueError(message)

    monkeypatch.setattr(simplejpeg, 'decode_jpeg', fail)


def test_a_header_the_fast_decoder_rejects_decodes_through_pillow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reject(HEADER_ERROR, monkeypatch)
    out = decode_image(_jpeg())
    assert out.pil.mode == 'RGB'
    assert out.pil.size == (24, 16)


def test_any_other_fast_decoder_error_still_fails_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reject('Premature end of JPEG file', monkeypatch)
    with pytest.raises(ImageDecodeError):
        decode_image(_jpeg())

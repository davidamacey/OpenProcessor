"""decode_image refuses an oversized image from its header, before any bitmap is allocated."""

from __future__ import annotations

import io

import cv2
import pytest
from PIL import Image

from src.utils import image_processing


def _jpeg(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new('L', size).save(buf, format='JPEG')
    return buf.getvalue()


def test_an_oversized_jpeg_is_refused_before_decode(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError('cv2.imdecode ran on an oversized image')

    monkeypatch.setattr(cv2, 'imdecode', boom)
    with pytest.raises(ValueError, match='larger than'):
        image_processing.decode_image(_jpeg((20000, 20000)), 'big.jpg')


def test_a_normal_jpeg_still_decodes() -> None:
    img = image_processing.decode_image(_jpeg((4000, 3000)), 'ok.jpg')
    assert img.shape[:2] == (3000, 4000)

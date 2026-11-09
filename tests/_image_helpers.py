"""Synthetic encoded images shared by the tests."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import cv2
from PIL import Image


if TYPE_CHECKING:
    import numpy as np


def make_jpeg(
    width: int = 320, height: int = 240, color: tuple[int, int, int] = (50, 80, 120)
) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), color).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def png_bytes(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', arr)
    assert ok
    return buf.tobytes()

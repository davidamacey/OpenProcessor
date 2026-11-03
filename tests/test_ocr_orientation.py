"""``/ocr`` decodes once, oriented, and keeps the alpha-to-white composite (GPU-3a, GPU-13).

``OcrService`` reports ``image_size`` from its own oriented decode, so the pixels sent to the
OCR pipeline must be that same frame. ``IMREAD_UNCHANGED`` ignores EXIF; ``IMREAD_COLOR``
honours it but drops alpha, so neither flag alone is right for a transparent PNG.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from src.clients import triton_ocr
from src.services import ocr_service as svc_mod
from src.services.ocr_service import OcrService


ORIENTATION_TAG = 0x0112


class _Response:
    def as_numpy(self, name: str) -> np.ndarray:
        return {
            'num_texts': np.array([0], dtype=np.int32),
            'text_boxes': np.zeros((128, 8), np.float32),
            'text_boxes_normalized': np.zeros((128, 4), np.float32),
            'texts': np.array([b''] * 128, dtype=object),
            'text_scores': np.zeros(128, np.float32),
            'rec_scores': np.zeros(128, np.float32),
        }[name]


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """Route ``OcrService`` through the real ``infer_ocr`` and capture what it sends."""
    captured: dict[str, np.ndarray] = {}

    def infer(_model: str, inputs: list, _outputs: list) -> _Response:
        dtypes = {'UINT8': np.uint8, 'INT32': np.int32}
        for inp in inputs:
            captured[inp.name()] = np.frombuffer(
                inp._get_content(), dtype=dtypes[inp.datatype()]
            ).reshape(inp.shape())
        return _Response()

    class _Client:
        def infer_ocr(self, image: Any) -> dict[str, Any]:
            return triton_ocr.infer_ocr(infer, image, lambda _n: 'UINT8')

    monkeypatch.setattr(svc_mod, 'get_sync_triton_client', lambda *_a, **_k: _Client())
    return captured


def _encode(img: Image.Image, fmt: str, orientation: int) -> bytes:
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = orientation
    buf = io.BytesIO()
    img.save(buf, format=fmt, exif=exif)
    return buf.getvalue()


def _gradient_rgb(w: int, h: int) -> Image.Image:
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, 1)
    return Image.fromarray(np.dstack([x, y, np.full_like(x, 128)]), 'RGB')


class TestOrientedFrame:
    def test_orientation_6_pixels_match_the_reported_image_size(
        self, sent: dict[str, np.ndarray]
    ) -> None:
        data = _encode(_gradient_rgb(200, 100), 'JPEG', 6)
        result = OcrService().extract_text(data)

        assert result['image_size'] == [200, 100]  # oriented: 100 wide, 200 high
        assert sent['orig_shape'].tolist() == [200, 100]
        assert sent['original_image'].shape == (3, 200, 100)

    def test_orientation_1_is_the_plain_decode(self, sent: dict[str, np.ndarray]) -> None:
        # Control: no rotation, so the oriented decode is the old decode.
        data = _encode(_gradient_rgb(200, 100), 'JPEG', 1)
        result = OcrService().extract_text(data)

        assert result['image_size'] == [100, 200]
        assert sent['orig_shape'].tolist() == [100, 200]
        plain = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(sent['original_image'].transpose(1, 2, 0), plain)


class TestAlpha:
    def test_transparent_png_composites_white_and_is_oriented(
        self, sent: dict[str, np.ndarray]
    ) -> None:
        rgba = Image.new('RGBA', (200, 100), (0, 0, 0, 0))
        rgba.paste((255, 0, 0, 255), (0, 0, 20, 20))  # opaque red top-left patch
        result = OcrService().extract_text(_encode(rgba, 'PNG', 6))

        assert result['image_size'] == [200, 100]
        img = sent['original_image'].transpose(1, 2, 0)  # HWC BGR, 200 high x 100 wide
        assert img.shape == (200, 100, 3)
        # Orientation 6 turns the top-left patch into the top-right corner.
        assert img[5, 95].tolist() == [0, 0, 255]
        assert img[100, 50].tolist() == [255, 255, 255]  # transparent: white, not black

    def test_transparent_png_without_rotation_still_composites_white(
        self, sent: dict[str, np.ndarray]
    ) -> None:
        # Control for the trap: IMREAD_COLOR alone would turn this black.
        rgba = Image.new('RGBA', (200, 100), (0, 0, 0, 0))
        OcrService().extract_text(_encode(rgba, 'PNG', 1))

        assert set(np.unique(sent['original_image'])) == {255}

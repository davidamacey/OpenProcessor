"""An OCR failure is an error, never "no text": ``/ocr/predict`` answers
502 (inference backend) or 422 (unusable image), and ``/analyze`` reports
``ocr_error`` instead of an empty ``ocr``."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client() -> TestClient:
    from src.main import app

    return TestClient(app)


@pytest.fixture
def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (10, 20, 30)).save(buf, format='JPEG')
    return buf.getvalue()


class _FailingTriton:
    def infer_ocr(self, _image_bytes: bytes) -> dict:
        raise RuntimeError('model ocr_pipeline failed: shape mismatch')


@pytest.fixture
def triton_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.ocr_service.get_triton_client', lambda *_a, **_k: _FailingTriton()
    )


def test_predict_is_502_when_inference_fails(
    client: TestClient, jpeg: bytes, triton_fails: None
) -> None:
    response = client.post('/ocr/predict', files={'image': ('t.jpg', jpeg, 'image/jpeg')})
    assert response.status_code == 502, response.text
    assert 'shape mismatch' in response.text


def test_predict_is_422_for_an_undecodable_image(client: TestClient) -> None:
    response = client.post(
        '/ocr/predict', files={'image': ('t.jpg', b'not an image', 'image/jpeg')}
    )
    assert response.status_code == 422, response.text


def test_predict_with_no_text_is_still_a_success(
    client: TestClient, jpeg: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _NoText:
        def infer_ocr(self, _image_bytes: bytes) -> dict:
            return {'num_texts': 0}

    monkeypatch.setattr('src.services.ocr_service.get_triton_client', lambda *_a, **_k: _NoText())
    response = client.post('/ocr/predict', files={'image': ('t.jpg', jpeg, 'image/jpeg')})
    assert response.status_code == 200
    assert response.json()['num_texts'] == 0


def test_analyze_ocr_helper_reports_a_failed_result_and_not_an_empty_one() -> None:
    from src.routers.analyze import _ocr_failure
    from src.services.ocr_service import OcrService

    failed = OcrService()._empty_result(error='boom', kind='inference')
    empty = OcrService()._empty_result(image_size=[8, 8])
    assert _ocr_failure(failed) == 'boom'
    assert _ocr_failure(empty) is None
    assert _ocr_failure(None) is None

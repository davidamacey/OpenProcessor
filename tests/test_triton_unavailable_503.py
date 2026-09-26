"""Core routes (/detect, /faces/*, /embed/*, /ocr/*, /analyze) must answer
``503`` with a ``Retry-After`` header when Triton is unreachable after
retries -- not a bare ``500`` indistinguishable from a code bug.

``RetryExhaustedError`` (``src/utils/retry.py``) is what
``TritonClient._infer_with_retry`` raises once ``max_retries`` gRPC
``UNAVAILABLE``/connection-refused attempts are exhausted. Each route's
service layer is monkeypatched to raise it directly -- this pins the
router -> centralized-handler wiring (``src/main.py``'s
``triton_unavailable_handler``), not the retry backoff itself (covered by
``src/utils/retry.py``'s own tests).

Uses the real app without the TestClient context manager (per
``tests/integration/test_request_id_propagation.py``'s established
pattern) so the Triton pool / OpenSearch lifespan never runs.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.utils.retry import RetryExhaustedError


@pytest.fixture
def client() -> TestClient:
    from src.main import app

    return TestClient(app)


@pytest.fixture
def tiny_jpeg_bytes() -> bytes:
    """A real, decodable 8x8 JPEG -- unlike a hand-rolled byte string, this
    survives PIL/cv2 decode so the request reaches the mocked service call
    instead of 400ing on an unreadable image first."""
    buf = io.BytesIO()
    Image.new('RGB', (8, 8), (10, 20, 30)).save(buf, format='JPEG')
    return buf.getvalue()


def _assert_503(response) -> None:
    assert response.status_code == 503
    assert response.headers.get('Retry-After')
    body = response.json()
    assert 'request_id' in body
    assert body['error_type'] == 'RetryExhaustedError'


def test_detect_returns_503_when_triton_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tiny_jpeg_bytes: bytes
) -> None:
    from src.routers import detect

    def boom(*_a, **_k):
        raise RetryExhaustedError('triton unavailable: all retries exhausted')

    monkeypatch.setattr(detect.inference_service, 'detect', boom)
    response = client.post('/detect', files={'image': ('t.jpg', tiny_jpeg_bytes, 'image/jpeg')})
    _assert_503(response)


def test_faces_detect_returns_503_when_triton_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tiny_jpeg_bytes: bytes
) -> None:
    from src.services.inference import InferenceService

    def boom(*_a, **_k):
        raise RetryExhaustedError('triton unavailable: all retries exhausted')

    monkeypatch.setattr(InferenceService, 'detect_faces', boom)
    response = client.post(
        '/faces/detect', files={'image': ('t.jpg', tiny_jpeg_bytes, 'image/jpeg')}
    )
    _assert_503(response)


def test_embed_image_returns_503_when_triton_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tiny_jpeg_bytes: bytes
) -> None:
    from src.services.inference import InferenceService

    def boom(*_a, **_k):
        raise RetryExhaustedError('triton unavailable: all retries exhausted')

    monkeypatch.setattr(InferenceService, 'encode_image', boom)
    response = client.post(
        '/embed/image', files={'image': ('t.jpg', tiny_jpeg_bytes, 'image/jpeg')}
    )
    _assert_503(response)


def test_ocr_predict_returns_503_when_triton_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tiny_jpeg_bytes: bytes
) -> None:
    from src.services.ocr_service import OcrService

    def boom(*_a, **_k):
        raise RetryExhaustedError('triton unavailable: all retries exhausted')

    monkeypatch.setattr(OcrService, 'extract_text', boom)
    response = client.post(
        '/ocr/predict', files={'image': ('t.jpg', tiny_jpeg_bytes, 'image/jpeg')}
    )
    _assert_503(response)


def test_analyze_returns_503_when_triton_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tiny_jpeg_bytes: bytes
) -> None:
    """``/analyze`` fans out to YOLO+CLIP (direct ``TritonClient``), faces
    (``InferenceService.detect_faces``) and OCR (``OcrService.extract_text``)
    in parallel -- any of the three raising ``RetryExhaustedError`` must
    surface as 503, not get swallowed into a per-component 'status': 'error'
    dict (the pre-fix behavior in ``src/services/inference.py``)."""
    from src.clients import triton_client as triton_client_mod

    class _BoomClient:
        def infer_yolo_clip_cpu(self, *_a, **_k):
            raise RetryExhaustedError('triton unavailable: all retries exhausted')

    monkeypatch.setattr(triton_client_mod, 'get_triton_client', lambda *_a, **_k: _BoomClient())
    response = client.post('/analyze', files={'image': ('t.jpg', tiny_jpeg_bytes, 'image/jpeg')})
    _assert_503(response)


def test_triton_unavailable_is_not_the_generic_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tiny_jpeg_bytes: bytes
) -> None:
    """Regression guard: before this fix, RetryExhaustedError fell into
    each router's broad ``except Exception`` and came back as a bare 500
    with no Retry-After -- indistinguishable from a genuine code bug."""
    from src.routers import detect

    def boom(*_a, **_k):
        raise RetryExhaustedError('triton unavailable: all retries exhausted')

    monkeypatch.setattr(detect.inference_service, 'detect', boom)
    response = client.post('/detect', files={'image': ('t.jpg', tiny_jpeg_bytes, 'image/jpeg')})
    assert response.status_code != 500

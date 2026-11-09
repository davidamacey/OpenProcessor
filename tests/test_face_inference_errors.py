"""A failed face inference raises a typed error the app maps; it never comes back as a
``status=error`` dict that a caller must remember to check (analyze and visual search did not)."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tritonclient.utils import InferenceServerException

import src.services.inference as inf
from src.core.exceptions import RequestRejectedError
from src.utils.upstream_errors import UpstreamError


class _Face:
    def __init__(self, result: dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc

    def recognize(self, *_a: Any, **_k: Any) -> dict[str, Any]:
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


class _Triton:
    def infer_yolo_clip_cpu(self, _b: bytes) -> dict[str, Any]:
        return {'image_embedding': [1.0]}

    def format_detections(self, *_a: Any, **_k: Any) -> list:
        return []


def _service(monkeypatch: pytest.MonkeyPatch, face: _Face) -> inf.InferenceService:
    monkeypatch.setattr(inf, 'get_fast_face_client', lambda *_a, **_k: face)
    monkeypatch.setattr(inf, 'get_sync_triton_client', lambda *_a, **_k: _Triton())
    svc = inf.InferenceService.__new__(inf.InferenceService)
    svc.settings = SimpleNamespace(triton_url='x')  # type: ignore[assignment]
    return svc


DECODE_FAILED = {'status': 'error', 'error': 'Failed to decode image'}


@pytest.mark.parametrize('method', ['detect_faces', 'recognize_faces'])
def test_an_undecodable_image_is_a_400(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    svc = _service(monkeypatch, _Face(DECODE_FAILED))
    with pytest.raises(RequestRejectedError) as caught:
        getattr(svc, method)(b'x')
    assert caught.value.status_code == 400
    assert caught.value.detail == 'Failed to decode image'


@pytest.mark.parametrize('method', ['detect_faces', 'recognize_faces'])
def test_a_failed_triton_call_raises_a_typed_upstream_error(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    svc = _service(monkeypatch, _Face(exc=InferenceServerException('model not ready')))
    with pytest.raises(UpstreamError) as caught:
        getattr(svc, method)(b'x')
    assert caught.value.reason == 'face_inference_failed'


def test_a_bug_in_the_face_client_is_not_reported_as_an_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = _service(monkeypatch, _Face(exc=KeyError('bug')))
    with pytest.raises(KeyError):
        svc.detect_faces(b'x')


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (8, 8), (10, 20, 30)).save(buf, format='JPEG')
    return buf.getvalue()


@pytest.mark.parametrize(
    ('face', 'status'),
    [
        (_Face(DECODE_FAILED), 400),
        (_Face(exc=InferenceServerException('model not ready')), 503),
    ],
)
def test_the_face_routes_map_the_typed_errors(
    monkeypatch: pytest.MonkeyPatch, face: _Face, status: int
) -> None:
    from src.main import app

    monkeypatch.setattr(inf, 'get_fast_face_client', lambda *_a, **_k: face)
    client = TestClient(app)
    for path in ('/faces/detect', '/faces/recognize'):
        response = client.post(path, files={'image': ('t.jpg', _jpeg(), 'image/jpeg')})
        assert response.status_code == status, (path, response.text)

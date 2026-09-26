"""The ``ocr_pipeline`` BLS (``models/ocr_pipeline/1/model.py``) emits
``-1.0`` in its ``rec_scores`` output tensor for a text crop whose
recognition inference itself failed -- distinguishable from a real CTC
confidence, always in ``[0, 1]``. That sentinel must never leave
``TritonClient.infer_ocr`` (or the curation cascade's independent BLS
parse in ``cascade_detect.py``): every served field gets ``null`` plus an
explicit failure reason instead.

Traced consumers (see the class-id-followups branch notes / PR description
for the full map): ``TritonClient.infer_ocr`` -> ``OcrService.extract_text``
-> ``/ocr/predict`` (``TextRegion.rec_score``/``rec_error``), ``/analyze``
(``OcrResult.rec_scores``/``rec_errors``), and OpenSearch's ``rec_score``
write path. The curation ``item_text_lines[].confidence`` /
``region_text_confidence`` wire fields are fed by an independent parse in
``cascade_detect.py`` that already drops a sentinel line (its text is
always ``''`` on failure) -- pinned here too as a defense-in-depth guard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


# =============================================================================
# 1. TritonClient.infer_ocr -- the earliest chokepoint
# =============================================================================


class _FakeResponse:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    def as_numpy(self, name: str) -> np.ndarray:
        return self._arrays[name]


def _fake_ocr_response(rec_scores: list[float]) -> _FakeResponse:
    n = len(rec_scores)
    return _FakeResponse(
        {
            'num_texts': np.array([n], dtype=np.int32),
            'text_boxes': np.zeros((n, 8), dtype=np.float32),
            'text_boxes_normalized': np.zeros((n, 4), dtype=np.float32),
            'texts': np.array([f't{i}'.encode() for i in range(n)], dtype=object),
            'text_scores': np.array([0.9] * n, dtype=np.float32),
            'rec_scores': np.array(rec_scores, dtype=np.float32),
        }
    )


@pytest.fixture
def triton_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.clients import triton_client as tc_mod

    monkeypatch.setattr(
        tc_mod.TritonClientManager, 'get_sync_client', staticmethod(lambda *_a, **_k: object())
    )
    return tc_mod.TritonClient()


def test_infer_ocr_replaces_negative_sentinel_with_none_and_reason(
    triton_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    monkeypatch.setattr(
        triton_client, '_infer_with_retry', lambda *_a, **_k: _fake_ocr_response([0.9, -1.0, 0.5])
    )
    image_bytes = cv2.imencode('.jpg', np.zeros((32, 32, 3), dtype=np.uint8))[1].tobytes()

    result = triton_client.infer_ocr(image_bytes)

    assert result['rec_scores'][0] == pytest.approx(0.9, abs=1e-5)
    assert result['rec_scores'][1] is None
    assert result['rec_scores'][2] == pytest.approx(0.5, abs=1e-5)
    assert result['rec_errors'] == [None, 'recognition_failed', None]
    # The sentinel value itself must not survive anywhere in the output.
    assert -1.0 not in [s for s in result['rec_scores'] if s is not None]


def test_infer_ocr_all_success_has_no_errors(
    triton_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    monkeypatch.setattr(
        triton_client, '_infer_with_retry', lambda *_a, **_k: _fake_ocr_response([0.9, 0.95])
    )
    image_bytes = cv2.imencode('.jpg', np.zeros((32, 32, 3), dtype=np.uint8))[1].tobytes()

    result = triton_client.infer_ocr(image_bytes)

    assert result['rec_scores'][0] == pytest.approx(0.9, abs=1e-5)
    assert result['rec_scores'][1] == pytest.approx(0.95, abs=1e-5)
    assert result['rec_errors'] == [None, None]


# =============================================================================
# 2. OcrService.extract_text -- filtering never lets None through as a
#    fake low score, even under a caller-supplied negative threshold.
# =============================================================================


@pytest.fixture
def tiny_jpeg_bytes() -> bytes:
    import cv2

    return cv2.imencode('.jpg', np.zeros((32, 32, 3), dtype=np.uint8))[1].tobytes()


def _patch_infer_ocr(monkeypatch: pytest.MonkeyPatch, sanitized_result: dict[str, Any]) -> None:
    from src.services import ocr_service as svc_mod

    class _FakeClient:
        def infer_ocr(self, _image_bytes: bytes) -> dict[str, Any]:
            return sanitized_result

    monkeypatch.setattr(svc_mod, 'get_triton_client', lambda *_a, **_k: _FakeClient())


def test_extract_text_filters_out_none_rec_score_even_under_negative_threshold(
    monkeypatch: pytest.MonkeyPatch, tiny_jpeg_bytes: bytes
) -> None:
    from src.services.ocr_service import OcrService

    _patch_infer_ocr(
        monkeypatch,
        {
            'num_texts': 2,
            'texts': ['good', 'bad'],
            'text_boxes': [[0, 0, 1, 0, 1, 1, 0, 1]] * 2,
            'text_boxes_normalized': [[0, 0, 1, 1]] * 2,
            'text_scores': [0.9, 0.9],
            'rec_scores': [0.9, None],
            'rec_errors': [None, 'recognition_failed'],
        },
    )
    # A caller-supplied negative threshold must not defeat the None guard --
    # unlike a numeric comparison, `None >= -5.0` is never evaluated.
    service = OcrService(min_det_score=0.0, min_rec_score=-5.0)

    result = service.extract_text(tiny_jpeg_bytes, filter_by_score=True)

    assert result['texts'] == ['good']
    assert result['rec_scores'] == [0.9]
    assert None not in result['rec_scores']
    assert all(s >= 0.0 for s in result['rec_scores'])


def test_extract_text_unfiltered_still_serves_none_not_negative_one(
    monkeypatch: pytest.MonkeyPatch, tiny_jpeg_bytes: bytes
) -> None:
    from src.services.ocr_service import OcrService

    _patch_infer_ocr(
        monkeypatch,
        {
            'num_texts': 1,
            'texts': ['bad'],
            'text_boxes': [[0, 0, 1, 0, 1, 1, 0, 1]],
            'text_boxes_normalized': [[0, 0, 1, 1]],
            'text_scores': [0.9],
            'rec_scores': [None],
            'rec_errors': ['recognition_failed'],
        },
    )
    service = OcrService()

    result = service.extract_text(tiny_jpeg_bytes, filter_by_score=False)

    assert result['rec_scores'] == [None]
    assert result['rec_errors'] == ['recognition_failed']
    assert -1.0 not in result['rec_scores']


# =============================================================================
# 3. /ocr/predict -- served field is null + rec_error, never negative
# =============================================================================


def test_ocr_predict_serves_null_rec_score_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from src.services.ocr_service import OcrService

    def fake_extract_text(self, _image_bytes: bytes, filter_by_score: bool = True) -> dict:
        del self, filter_by_score
        return {
            'status': 'success',
            'texts': ['bad'],
            'boxes': [[0, 0, 1, 0, 1, 1, 0, 1]],
            'boxes_normalized': [[0, 0, 1, 1]],
            'det_scores': [0.9],
            'rec_scores': [None],
            'rec_errors': ['recognition_failed'],
            'num_texts': 1,
            'image_size': [8, 8],
        }

    monkeypatch.setattr(OcrService, 'extract_text', fake_extract_text)

    from src.main import app

    client = TestClient(app)
    response = client.post(
        '/ocr/predict',
        params={'filter_by_score': False},
        files={'image': ('t.jpg', b'\xff\xd8\xff\xe0fake', 'image/jpeg')},
    )
    assert response.status_code == 200
    body = response.json()
    [region] = body['regions']
    assert region['rec_score'] is None
    assert region['rec_error'] == 'recognition_failed'


def test_no_response_field_in_ocr_predict_can_be_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard over the raw JSON body: nothing numeric is < 0."""
    from fastapi.testclient import TestClient

    from src.services.ocr_service import OcrService

    def fake_extract_text(self, _image_bytes: bytes, filter_by_score: bool = True) -> dict:
        del self, filter_by_score
        return {
            'status': 'success',
            'texts': ['good', 'bad'],
            'boxes': [[0, 0, 1, 0, 1, 1, 0, 1]] * 2,
            'boxes_normalized': [[0, 0, 1, 1]] * 2,
            'det_scores': [0.9, 0.9],
            'rec_scores': [0.9, None],
            'rec_errors': [None, 'recognition_failed'],
            'num_texts': 2,
            'image_size': [8, 8],
        }

    monkeypatch.setattr(OcrService, 'extract_text', fake_extract_text)

    from src.main import app

    client = TestClient(app)
    response = client.post(
        '/ocr/predict',
        params={'filter_by_score': False},
        files={'image': ('t.jpg', b'\xff\xd8\xff\xe0fake', 'image/jpeg')},
    )
    assert response.status_code == 200

    def _walk(node: Any) -> None:
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            assert node >= 0, f'negative value leaked into response: {node!r}'
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(response.json())


# =============================================================================
# 4. cascade_detect._parse_ocr_pipeline_result -- curation's independent
#    BLS parse (item_text_lines[].confidence / region_text_confidence).
# =============================================================================


def test_cascade_parse_drops_sentinel_line_even_with_nonempty_text() -> None:
    """Defense in depth: a sentinel line is dropped by the empty-text check
    in practice (the BLS always pairs -1.0 with ''), but this pins the
    explicit `rec_score < 0` guard for a future BLS build that might not."""
    from src.services.detection.cascade_detect import _parse_ocr_pipeline_result

    class _FakeResult:
        def as_numpy(self, name: str) -> np.ndarray:
            arrays = {
                'num_texts': np.array([2], dtype=np.int32),
                'text_boxes_normalized': np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
                'texts': np.array([b'good', b'oops'], dtype=object),
                'text_scores': np.array([0.9, 0.9], dtype=np.float32),
                'rec_scores': np.array([0.9, -1.0], dtype=np.float32),
            }
            return arrays[name]

    lines = _parse_ocr_pipeline_result(_FakeResult())

    assert [ln.text for ln in lines] == ['good']
    assert all(ln.score >= 0.0 for ln in lines)

"""Behavior of ``POST /analyze`` and ``POST /analyze/batch`` with every backend faked.

The fakes sit at seams that do not depend on how the handlers are organised: the
Triton client factory, ``InferenceService``, ``OcrService`` and the class-name lookup.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.utils.retry import RetryExhaustedError


IMG_W, IMG_H = 200, 100


@pytest.fixture
def jpeg() -> bytes:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='JPEG')
    return buf.getvalue()


def _yolo_result(num_dets: int = 3) -> dict[str, Any]:
    boxes = np.array(
        [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32
    )
    return {
        'num_dets': num_dets,
        'boxes': boxes[:num_dets],
        'scores': np.array([0.9, 0.1, 0.5], dtype=np.float32)[:num_dets],
        'classes': np.array([0, 1, 2])[:num_dets],
        'orig_shape': (IMG_H, IMG_W),
        'input_size': 640,
        'image_embedding': np.array([3.0, 4.0]),
    }


class FakeTriton:
    def __init__(self, yolo: dict[str, Any], fail_calls: tuple[int, ...] = ()) -> None:
        self.yolo = yolo
        self.fail_calls = fail_calls
        self.yolo_calls = 0
        self.clip_batches: list[np.ndarray] = []
        self.clip_error: Exception | None = None
        self.yolo_error: Exception | None = None

    def infer_yolo_clip_cpu(self, _image_bytes: bytes) -> dict[str, Any]:
        self.yolo_calls += 1
        if self.yolo_error is not None:
            raise self.yolo_error
        if self.yolo_calls in self.fail_calls:
            raise RuntimeError(f'triton exploded on call {self.yolo_calls}')
        return self.yolo

    def infer_mobileclip_batch(self, batch: np.ndarray) -> np.ndarray:
        self.clip_batches.append(batch)
        if self.clip_error is not None:
            raise self.clip_error
        n = len(batch)
        return np.arange(n * 512, dtype=np.float32).reshape(n, 512)


class FakeFaces:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.confidences: list[float] = []

    def detect_faces(self, _image_bytes: bytes, confidence: float = 0.5) -> dict[str, Any]:
        self.confidences.append(confidence)
        return self.result


class FakeOcr:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls = 0

    def extract_text(self, _image_bytes: bytes, filter_by_score: bool = False) -> dict[str, Any]:
        assert filter_by_score is True
        self.calls += 1
        return self.result


def _faces_result(embeddings: int = 1) -> dict[str, Any]:
    faces = [
        {
            'box': [0.1, 0.1, 0.2, 0.2],
            'landmarks': [0.0] * 10,
            'score': 0.99,
            'quality': 0.7,
        },
        {'box': [0.3, 0.3, 0.4, 0.4], 'landmarks': [1.0] * 10, 'score': 0.8},
    ]
    return {
        'status': 'success',
        'num_faces': 2,
        'faces': faces,
        'embeddings': [np.full(4, float(i)) for i in range(embeddings)],
    }


OCR_OK = {
    'status': 'success',
    'texts': ['ab', 'cd'],
    'boxes': [[0.0] * 8],
    'boxes_normalized': [[0.0, 0.0, 1.0, 1.0]],
    'det_scores': [0.9],
    'rec_scores': [0.8, None],
    'rec_errors': [None, 'recognition_failed'],
    'num_texts': 2,
}


class Backends:
    def __init__(self, triton: FakeTriton, faces: FakeFaces, ocr: FakeOcr) -> None:
        self.triton = triton
        self.faces = faces
        self.ocr = ocr


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch) -> Backends:
    from src.services import analyze_pipeline, ocr_service

    b = Backends(FakeTriton(_yolo_result()), FakeFaces(_faces_result()), FakeOcr(OCR_OK))
    monkeypatch.setattr(
        'src.clients.triton_client.get_sync_triton_client', lambda *_a, **_k: b.triton
    )
    monkeypatch.setattr(analyze_pipeline, 'InferenceService', lambda: b.faces)
    monkeypatch.setattr(ocr_service, 'OcrService', lambda: b.ocr)
    monkeypatch.setattr(
        'src.utils.class_names.get_class_names', lambda _model: {0: 'cat', 1: 'dog'}
    )
    analyze_pipeline.get_inference_service.cache_clear()
    ocr_service.get_ocr_service.cache_clear()
    yield b
    analyze_pipeline.get_inference_service.cache_clear()
    ocr_service.get_ocr_service.cache_clear()


@pytest.fixture
def client() -> TestClient:
    from src.main import app

    return TestClient(app, raise_server_exceptions=False)


def _post(client: TestClient, jpeg: bytes, **params: Any):
    return client.post(
        '/analyze', files={'image': ('a.jpg', jpeg, 'image/jpeg')}, params=params or None
    )


def _batch(client: TestClient, jpeg: bytes, count: int = 1, **params: Any):
    files = [('images', (f'img{i}.jpg', jpeg, 'image/jpeg')) for i in range(count)]
    return client.post('/analyze/batch', files=files, params=params or None)


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


def test_single_default_reports_filtered_detections_faces_ocr_and_timing(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    body = _post(client, jpeg).json()
    assert body['status'] == 'success'
    assert body['image'] == {'width': IMG_W, 'height': IMG_H}
    # score 0.1 is below the default confidence of 0.25
    assert [d['class_id'] for d in body['detections']] == [0, 2]
    assert body['num_detections'] == 2
    assert body['detections'][0]['class_name'] == 'cat'
    assert body['detections'][1]['class_name'] == 'class_2'
    assert body['detections'][0]['box'] == pytest.approx([0.1, 0.1, 0.5, 0.5])
    assert body['detections'][0]['embedding'] is None
    assert body['num_faces'] == 2
    assert body['faces'][0]['quality'] == pytest.approx(0.7)
    assert body['faces'][1]['quality'] is None
    assert [f['embedding'] for f in body['faces']] == [None, None]
    assert body['global_embedding'] is None
    assert body['embedding_norm'] is None
    assert body['ocr']['texts'] == ['ab', 'cd']
    assert body['ocr']['full_text'] == 'ab cd'
    assert body['ocr']['num_texts'] == 2
    assert body['ocr']['rec_scores'] == [0.8, None]
    assert body['ocr_error'] is None
    assert set(body['timing']) == {'yolo_clip_ms', 'faces_ms', 'ocr_ms'}
    assert backends.faces.confidences == [0.5]
    assert backends.triton.clip_batches == []


def test_single_confidence_params_reach_the_filters(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    body = _post(client, jpeg, confidence=0.05, face_confidence=0.8).json()
    assert [d['class_id'] for d in body['detections']] == [0, 1, 2]
    assert backends.faces.confidences == [0.8]
    body = _post(client, jpeg, confidence=0.95).json()
    assert body['detections'] == []
    assert body['num_detections'] == 0


def test_single_embeddings_use_one_crop_batch_with_zero_rows_for_empty_boxes(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    body = _post(client, jpeg, include_embedding='true', include_face_embeddings='true').json()
    assert body['global_embedding'] == [3.0, 4.0]
    assert body['embedding_norm'] == pytest.approx(5.0)
    assert len(backends.triton.clip_batches) == 1
    batch = backends.triton.clip_batches[0]
    assert batch.shape == (2, 3, 256, 256)
    assert batch[0].any()
    assert not batch[1].any()  # the zero-area box gets a zero placeholder
    assert body['detections'][0]['embedding'][:2] == [0.0, 1.0]
    assert body['detections'][1]['embedding'][:2] == [512.0, 513.0]
    # one embedding for two faces: the second has none
    assert body['faces'][0]['embedding'] == [0.0, 0.0, 0.0, 0.0]
    assert body['faces'][1]['embedding'] is None
    assert set(body['timing']) == {'yolo_clip_ms', 'faces_ms', 'ocr_ms', 'crop_embeddings_ms'}


def test_single_a_failed_crop_batch_leaves_detections_without_embeddings(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.triton.clip_error = RuntimeError('clip down')
    response = _post(client, jpeg, include_embedding='true')
    assert response.status_code == 200
    body = response.json()
    assert [d['embedding'] for d in body['detections']] == [None, None]
    assert body['global_embedding'] == [3.0, 4.0]
    assert 'crop_embeddings_ms' not in body['timing']


def test_single_without_detections_skips_the_crop_batch(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.triton.yolo = _yolo_result(num_dets=0)
    body = _post(client, jpeg, include_embedding='true').json()
    assert body['detections'] == []
    assert backends.triton.clip_batches == []


def test_single_ocr_disabled_never_calls_ocr(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    body = _post(client, jpeg, enable_ocr='false').json()
    assert body['ocr'] is None
    assert body['ocr_error'] is None
    assert 'ocr_ms' not in body['timing']
    assert backends.ocr.calls == 0


def test_single_a_failed_ocr_is_reported_not_hidden(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.ocr.result = {'status': 'error', 'error': 'ocr model down'}
    body = _post(client, jpeg).json()
    assert body['ocr'] is None
    assert body['ocr_error'] == 'ocr model down'
    backends.ocr.result = {'status': 'error'}
    assert _post(client, jpeg).json()['ocr_error'] == 'OCR failed'


def test_single_a_failed_face_stage_yields_no_faces(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.faces.result = {'status': 'error', 'num_faces': 2, 'faces': _faces_result()['faces']}
    body = _post(client, jpeg).json()
    assert body['faces'] == []
    assert body['num_faces'] == 0


def test_single_unknown_image_size_omits_image(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.triton.yolo = {**_yolo_result(), 'orig_shape': (0, 0)}
    body = _post(client, jpeg).json()
    assert body['image'] is None


def test_single_empty_upload_is_a_400(client: TestClient, backends: Backends) -> None:
    response = client.post('/analyze', files={'image': ('a.jpg', b'', 'image/jpeg')})
    assert response.status_code == 400
    assert response.json()['detail'] == 'Empty image file'


def test_single_unexpected_failure_is_a_500_with_the_message(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.triton.yolo_error = ValueError('bad tensor')
    response = _post(client, jpeg)
    assert response.status_code == 500
    assert response.json()['detail'] == 'Analysis failed: bad tensor'


def test_single_retry_exhaustion_passes_through_to_the_triton_handler(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.triton.yolo_error = RetryExhaustedError('triton gone')
    response = _post(client, jpeg)
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /analyze/batch
# ---------------------------------------------------------------------------


def test_batch_summary_counts_without_full_results(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    body = _batch(client, jpeg, count=2, enable_ocr='true').json()
    assert body['status'] == 'success'
    assert (body['total_images'], body['processed_images'], body['failed_images']) == (2, 2, 0)
    assert body['failures'] is None
    # the summary counts raw detections, before any confidence filter
    assert body['total_detections'] == 6
    assert body['total_faces'] == 4
    assert body['images_with_text'] == 2
    first = body['results'][0]
    assert first['filename'] == 'img0.jpg'
    assert first['image_index'] == 0
    assert first['image'] == {'width': IMG_W, 'height': IMG_H}
    assert first['num_detections'] == 3
    assert first['has_text'] is True
    assert first['detections'] is None
    assert first['faces'] is None
    assert first['ocr'] is None
    assert backends.faces.confidences == [0.5, 0.5]


def test_batch_ocr_is_off_by_default(client: TestClient, jpeg: bytes, backends: Backends) -> None:
    body = _batch(client, jpeg).json()
    assert body['images_with_text'] == 0
    assert body['results'][0]['has_text'] is False
    assert backends.ocr.calls == 0


def test_batch_full_results_carry_detections_faces_ocr_and_embeddings(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    body = _batch(
        client,
        jpeg,
        include_full_results='true',
        include_embedding='true',
        include_face_embeddings='true',
        enable_ocr='true',
    ).json()
    result = body['results'][0]
    assert [d['class_id'] for d in result['detections']] == [0, 2]
    assert result['detections'][0]['class_name'] == 'cat'
    assert result['detections'][0]['embedding'][:2] == [0.0, 1.0]
    assert result['detections'][1]['embedding'][:2] == [512.0, 513.0]
    batch = backends.triton.clip_batches[0]
    assert batch.shape == (2, 3, 256, 256)
    assert not batch[1].any()
    assert result['global_embedding'] == [3.0, 4.0]
    assert result['faces'][0]['embedding'] == [0.0, 0.0, 0.0, 0.0]
    assert result['faces'][1]['embedding'] is None
    assert result['ocr']['full_text'] == 'ab cd'
    assert result['ocr_error'] is None


def test_batch_full_results_confidence_filters_detections_only(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    body = _batch(client, jpeg, include_full_results='true', confidence=0.95).json()
    result = body['results'][0]
    assert result['detections'] == []
    assert result['num_detections'] == 3
    assert result['global_embedding'] is None


def test_batch_full_results_without_embedding_flags_make_no_clip_call(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    body = _batch(client, jpeg, include_full_results='true').json()
    assert backends.triton.clip_batches == []
    assert body['results'][0]['detections'][0]['embedding'] is None
    assert [f['embedding'] for f in body['results'][0]['faces']] == [None, None]


def test_batch_a_failed_crop_batch_does_not_fail_the_image(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.triton.clip_error = RuntimeError('clip down')
    body = _batch(client, jpeg, include_full_results='true', include_embedding='true').json()
    assert body['processed_images'] == 1
    assert [d['embedding'] for d in body['results'][0]['detections']] == [None, None]
    assert body['results'][0]['global_embedding'] == [3.0, 4.0]


def test_batch_full_results_report_a_failed_ocr(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.ocr.result = {'status': 'error', 'error': 'ocr model down'}
    body = _batch(client, jpeg, include_full_results='true', enable_ocr='true').json()
    result = body['results'][0]
    assert result['ocr'] is None
    assert result['ocr_error'] == 'ocr model down'
    assert result['has_text'] is False


def test_batch_without_faces_skips_face_parsing(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.faces.result = {'status': 'success', 'num_faces': 0, 'faces': []}
    body = _batch(client, jpeg, include_full_results='true').json()
    assert body['results'][0]['faces'] == []
    assert body['total_faces'] == 0


def test_batch_one_failing_image_is_listed_and_the_rest_continue(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.triton.fail_calls = (1,)
    body = _batch(client, jpeg, count=2).json()
    assert body['status'] == 'success'
    assert (body['processed_images'], body['failed_images']) == (1, 1)
    assert body['failures'] == [
        {'filename': 'img0.jpg', 'index': 0, 'error': 'triton exploded on call 1'}
    ]
    assert body['results'][0]['image_index'] == 1


def test_batch_all_images_failing_is_all_failed(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    backends.triton.fail_calls = (1, 2)
    body = _batch(client, jpeg, count=2).json()
    assert body['status'] == 'all_failed'
    assert body['results'] == []
    assert body['failed_images'] == 2
    assert body['total_detections'] == 0


def test_batch_an_empty_file_is_a_failure_entry(client: TestClient, backends: Backends) -> None:
    response = client.post('/analyze/batch', files=[('images', ('e.jpg', b'', 'image/jpeg'))])
    body = response.json()
    assert body['status'] == 'all_failed'
    assert body['failures'] == [{'filename': 'e.jpg', 'index': 0, 'error': 'Empty image file'}]
    assert backends.triton.yolo_calls == 0


def test_batch_more_than_32_images_is_a_400(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    response = _batch(client, jpeg, count=33)
    assert response.status_code == 400
    assert response.json()['detail'] == 'Batch size 33 exceeds maximum 32'
    assert backends.triton.yolo_calls == 0


def test_batch_with_no_images_is_an_empty_success(backends: Backends) -> None:
    from src.routers.analyze import analyze_batch

    result = analyze_batch(
        images=[],
        include_embedding=False,
        include_face_embeddings=False,
        enable_ocr=False,
        include_full_results=False,
        confidence=0.25,
    )
    assert result.status == 'success'
    assert (result.total_images, result.processed_images, result.failed_images) == (0, 0, 0)
    assert result.results == []
    assert result.failures is None


def test_batch_an_image_failing_while_parsing_full_results_still_counts_in_the_totals(
    client: TestClient, jpeg: bytes, backends: Backends
) -> None:
    """The summary counts are added before the full results are built, so an image that
    then fails still contributes to them (and is listed as a failure)."""
    backends.faces.result = {'status': 'success', 'num_faces': 1, 'faces': [{}]}
    body = _batch(client, jpeg, include_full_results='true', enable_ocr='true').json()
    assert body['status'] == 'all_failed'
    assert body['failures'] == [{'filename': 'img0.jpg', 'index': 0, 'error': "'box'"}]
    assert body['total_detections'] == 3
    assert body['total_faces'] == 1
    assert body['images_with_text'] == 1

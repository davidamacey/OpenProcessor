"""A face response whose landmark list is shorter than its embeddings fails only the
affected item, with a stable reason; it never raises out of the batch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from src.services.visual_search.batch_documents import build_documents
from src.services.visual_search.ingest_batch import IngestBatchMixin
from src.services.visual_search.ingest_single import IngestSingleMixin


if TYPE_CHECKING:
    import pytest


def _faces(n: int, landmarks: int) -> dict[str, Any]:
    return {
        'num_faces': n,
        'face_embeddings': [[0.1, 0.2]] * n,
        'face_boxes': [[0.1, 0.1, 0.2, 0.2]] * n,
        'face_scores': [0.9] * n,
        'face_quality': [0.8] * n,
        'face_landmarks': [[0.5] * 10] * landmarks,
    }


def _build(results: list[dict]) -> tuple[list, list, list]:
    items = [(b'x', f'id-{i}', None) for i in range(len(results))]
    return build_documents(
        results, items, [f'h{i}' for i in range(len(results))], list(range(len(results))), 't'
    )


def test_short_landmarks_fail_only_that_image() -> None:
    good = {'global_embedding': [1.0], 'orig_shape': (10, 10), **_faces(2, 2)}
    bad = {'global_embedding': [1.0], 'orig_shape': (10, 10), **_faces(2, 1)}
    docs, faces, errors = _build([good, bad])
    assert [d['image_id'] for d in docs] == ['id-0']
    assert [f['image_id'] for f in faces] == ['id-0', 'id-0']
    assert errors == [
        {
            'image_id': 'id-1',
            'error': 'face_landmarks_length_mismatch: 1 landmark entries for 2 faces',
        }
    ]


def test_absent_landmarks_are_kept_as_empty() -> None:
    res = {'global_embedding': [1.0], 'orig_shape': (10, 10), **_faces(1, 0)}
    docs, faces, errors = _build([res])
    assert len(docs) == 1
    assert not errors
    assert faces[0]['landmarks'] == []


def test_single_image_path_skips_only_the_face_without_landmarks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    indexed: list[str] = []

    class FakeOS:
        async def index_face(self, **kw: Any) -> None:
            indexed.append(kw['face_id'])

    svc = SimpleNamespace(opensearch=FakeOS())
    with caplog.at_level('WARNING'):
        n = asyncio.run(IngestSingleMixin._index_faces(svc, _faces(2, 1), 'img', '/p'))  # type: ignore[arg-type]
    assert n == 1
    assert indexed == ['img_face_0']
    assert 'img_face_1: face_landmarks_length_mismatch' in caplog.text


def test_ingest_faces_reports_the_reason() -> None:
    class FakeClient:
        def recognize(self, *_a: Any, **_k: Any) -> dict[str, Any]:
            return {'status': 'success', **_faces(2, 1)}

    import src.clients.fast_face_client as ffc

    orig = ffc.get_fast_face_client
    ffc.get_fast_face_client = lambda *_a, **_k: FakeClient()  # type: ignore[assignment]
    try:
        svc = SimpleNamespace(opensearch=None)
        out = asyncio.run(IngestBatchMixin.ingest_faces(svc, b'x', 'img'))  # type: ignore[arg-type]
    finally:
        ffc.get_fast_face_client = orig
    assert out['status'] == 'error'
    assert out['error'].startswith('face_landmarks_length_mismatch')


def _inference_service(monkeypatch: pytest.MonkeyPatch, faces: dict[str, Any]) -> Any:
    import src.services.inference as inf

    class FakeFace:
        def recognize(self, *_a: Any, **_k: Any) -> dict[str, Any]:
            return {'status': 'success', 'orig_shape': (10, 10), **faces}

    class FakeTriton:
        def infer_yolo_clip_cpu(self, _b: bytes) -> dict[str, Any]:
            return {'image_embedding': [1.0]}

        def format_detections(self, *_a: Any, **_k: Any) -> list:
            return []

    monkeypatch.setattr(inf, 'get_fast_face_client', lambda *_a, **_k: FakeFace())
    monkeypatch.setattr(inf, 'get_sync_triton_client', lambda *_a, **_k: FakeTriton())
    svc = inf.InferenceService.__new__(inf.InferenceService)
    svc.settings = SimpleNamespace(triton_url='x')
    return svc


def test_detect_faces_reports_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _inference_service(monkeypatch, _faces(2, 1))
    out = svc.detect_faces(b'x')
    assert out['status'] == 'error'
    assert out['error'].startswith('face_landmarks_length_mismatch')
    assert out['faces'] == []


def test_recognize_faces_reports_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _inference_service(monkeypatch, _faces(2, 1))
    out = svc.recognize_faces(b'x')
    assert out['status'] == 'error'
    assert out['error'].startswith('face_landmarks_length_mismatch')
    assert out['faces'] == []

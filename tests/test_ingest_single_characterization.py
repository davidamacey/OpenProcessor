"""Pins ``VisualSearchService.ingest_image`` across its branches: the response, what is
handed to OpenSearch (global/box documents, faces), which pipelines run for each flag
combination, and the failure paths.

The golden file was captured from the implementation before ``ingest_image`` was
decomposed. Triton, the face client, OCR and OpenSearch are scripted fakes; the hash is
the real imohash of the scripted bytes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import imohash
import numpy as np
import pytest

from src.services.visual_search import VisualSearchService


GOLDEN = Path(__file__).parent / 'golden' / 'ingest_single_golden.json'
DATA = b'single-image-bytes'


def _hash(data: bytes) -> str:
    return imohash.hashfileobject(io.BytesIO(data)).hex()


YOLO: dict[str, Any] = {
    'num_dets': 2,
    'image_embedding': [3.0, 4.0],
    'box_embeddings': [[0.5, 0.5], [0.1, 0.9]],
    'normalized_boxes': [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]],
    'classes': [2, 0],
    'scores': [0.9, 0.8],
    'image_width': 640,
    'image_height': 480,
}


def _faces(n: int, *, numpy: bool = False, drop: str | None = None) -> dict[str, Any]:
    box = np.array([0.1, 0.1, 0.2, 0.2]) if numpy else [0.1, 0.1, 0.2, 0.2]
    out: dict[str, Any] = {
        'num_faces': n,
        'face_embeddings': [[0.1, 0.2]] * n,
        'face_boxes': [box] * n,
        'face_scores': [0.99] * n,
        'face_quality': [0.7] * n,
        'face_landmarks': [np.ones(10) if numpy else [1.0] * 10] * n,
    }
    if drop:
        out[drop] = out[drop][: n - 1]
    return out


OCR_RESULT = {
    'status': 'success',
    'num_texts': 2,
    'texts': ['AB', 'CD'],
    'boxes': [[1, 2, 3, 4]] * 2,
    'boxes_normalized': [[0.1, 0.1, 0.2, 0.2]] * 2,
    'det_scores': [0.9, 0.9],
    'rec_scores': [0.8, 0.8],
}

SCENARIOS: dict[str, dict[str, Any]] = {
    'full': {'faces': _faces(2), 'dup': 'joined'},
    'duplicate_found': {'existing': True},
    'duplicate_ignored_when_not_skipping': {'existing': True, 'kwargs': {'skip_duplicates': False}},
    'opensearch_duplicate_check_raises': {'dup_check_raises': True},
    'detection_only': {
        'kwargs': {'enable_clip': False, 'enable_faces': False, 'enable_ocr': False},
    },
    'clip_only': {
        'kwargs': {'enable_detection': False, 'enable_faces': False, 'enable_ocr': False},
    },
    'faces_only': {
        'faces': _faces(2),
        'kwargs': {'enable_detection': False, 'enable_clip': False, 'enable_ocr': False},
    },
    'nothing_enabled': {
        'kwargs': {
            'enable_detection': False,
            'enable_clip': False,
            'enable_faces': False,
            'enable_ocr': False,
        },
    },
    'no_detections': {'yolo': {**YOLO, 'num_dets': 0}},
    'no_embedding_in_response': {'yolo': {'num_dets': 0}},
    'metadata_and_path': {'kwargs': {'metadata': {'k': 'v'}, 'image_path': '/p/x.jpg'}},
    'faces_numpy': {'faces': _faces(2, numpy=True)},
    'faces_short_scores_and_quality': {'faces': _faces(3, drop='face_scores')},
    'faces_short_quality': {'faces': _faces(3, drop='face_quality')},
    'faces_short_landmarks_skips_that_face': {'faces': _faces(2, drop='face_landmarks')},
    'face_with_empty_embedding': {
        'faces': {**_faces(2), 'face_embeddings': [[], [0.1, 0.2]]},
    },
    'face_index_raises_for_one': {'faces': _faces(3), 'face_index_fail': ['id-1_face_1']},
    'global_index_fails': {'ingest': {'global': False, 'vehicles': 0, 'people': 0, 'skipped': 0}},
    'ingest_errors_reported': {
        'ingest': {
            'global': True,
            'vehicles': 1,
            'people': 0,
            'skipped': 1,
            'errors': ['Global: x'],
        },
    },
    'near_duplicate_disabled': {'dup': 'joined', 'kwargs': {'detect_near_duplicates': False}},
    'near_duplicate_created': {'dup': 'created', 'kwargs': {'near_duplicate_threshold': 0.9}},
    'near_duplicate_raises': {'dup': 'raise'},
    'ocr_finds_text': {'ocr': OCR_RESULT},
    'ocr_no_text': {'ocr': {'status': 'success', 'num_texts': 0}},
    'ocr_disabled': {'ocr': OCR_RESULT, 'kwargs': {'enable_ocr': False}},
    'triton_raises': {'triton_raises': True},
    'face_client_raises': {'face_raises': True},
    'opensearch_ingest_raises': {'ingest_raises': True},
}


class FakeTriton:
    def __init__(self, sc: dict[str, Any]) -> None:
        self.yolo = sc.get('yolo', YOLO)
        self.raises = sc.get('triton_raises', False)
        self.calls = 0

    def infer_yolo_clip_cpu(self, data: bytes) -> dict[str, Any]:
        assert data == DATA
        self.calls += 1
        if self.raises:
            msg = 'triton down'
            raise RuntimeError(msg)
        return dict(self.yolo)


class FakeFace:
    def __init__(self, sc: dict[str, Any]) -> None:
        self.result = sc.get('faces', _faces(0))
        self.raises = sc.get('face_raises', False)
        self.calls: list[float] = []

    def recognize(self, data: bytes, confidence: float) -> dict[str, Any]:
        assert data == DATA
        self.calls.append(confidence)
        if self.raises:
            msg = 'face down'
            raise RuntimeError(msg)
        return self.result


class FakeOcr:
    def __init__(self, sc: dict[str, Any]) -> None:
        self.result = sc.get('ocr', {'status': 'success', 'num_texts': 0})
        self.calls = 0

    def extract_text(self, data: bytes, filter_by_score: bool) -> dict[str, Any]:
        assert data == DATA
        assert filter_by_score is True
        self.calls += 1
        return self.result

    def get_full_text(self, result: dict[str, Any]) -> str:
        return ' '.join(result['texts'])


class FakeOpenSearch:
    def __init__(self, sc: dict[str, Any]) -> None:
        self.sc = sc
        self.calls: list[tuple[str, Any]] = []

    async def check_duplicate_by_hash(self, image_hash: str) -> Any:
        self.calls.append(('check_duplicate_by_hash', image_hash))
        if self.sc.get('dup_check_raises'):
            msg = 'search down'
            raise RuntimeError(msg)
        if self.sc.get('existing'):
            return {'image_id': 'old-1', 'image_path': '/old.jpg'}
        return None

    async def ingest_image(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(('ingest_image', kw))
        if self.sc.get('ingest_raises'):
            msg = 'index down'
            raise RuntimeError(msg)
        return self.sc.get(
            'ingest', {'global': True, 'vehicles': 1, 'people': 1, 'skipped': 0, 'errors': []}
        )

    async def index_face(self, **kw: Any) -> None:
        self.calls.append(('index_face', kw))
        if kw['face_id'] in self.sc.get('face_index_fail', []):
            msg = 'face index down'
            raise RuntimeError(msg)

    async def index_ocr_results(self, **kw: Any) -> None:
        self.calls.append(('index_ocr_results', kw))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {'ndarray': value.tolist()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


async def run_scenario(name: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sc = SCENARIOS[name]
    triton, face, ocr, opensearch = FakeTriton(sc), FakeFace(sc), FakeOcr(sc), FakeOpenSearch(sc)
    monkeypatch.setattr('src.clients.triton_client.get_sync_triton_client', lambda url: triton)  # noqa: ARG005
    monkeypatch.setattr('src.clients.fast_face_client.get_fast_face_client', lambda url: face)  # noqa: ARG005
    monkeypatch.setattr('src.services.ocr_service.get_ocr_service', lambda: ocr)
    svc = VisualSearchService.__new__(VisualSearchService)
    svc.opensearch = opensearch  # type: ignore[assignment]
    svc.inference = None  # type: ignore[assignment]
    dup_calls: list[dict[str, Any]] = []

    async def assign(image_id: str, embedding: Any, threshold: float = 0.99) -> Any:
        dup_calls.append(
            {'image_id': image_id, 'embedding': _plain(embedding), 'threshold': threshold}
        )
        kind = sc.get('dup')
        if kind == 'raise':
            msg = 'dup service down'
            raise RuntimeError(msg)
        if kind == 'joined':
            return {'action': 'joined_group', 'group_id': 'g1'}
        if kind == 'created':
            return {'action': 'created_group', 'group_id': 'g2'}
        return None

    svc._assign_to_duplicate_group = assign  # type: ignore[method-assign]
    result = await svc.ingest_image(DATA, 'id-1', **sc.get('kwargs', {}))
    return _plain(
        {
            'result': result,
            'opensearch': opensearch.calls,
            'dup_calls': dup_calls,
            'triton_calls': triton.calls,
            'face_calls': face.calls,
            'ocr_calls': ocr.calls,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('name', sorted(SCENARIOS))
async def test_ingest_image_matches_golden(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    got = await run_scenario(name, monkeypatch)
    assert got == json.loads(GOLDEN.read_text())[name]


def test_golden_file_has_exactly_the_scenarios() -> None:
    assert set(json.loads(GOLDEN.read_text())) == set(SCENARIOS)


def test_scripted_hash_is_the_real_imohash() -> None:
    golden = json.loads(GOLDEN.read_text())
    assert golden['full']['result']['imohash'] == _hash(DATA)

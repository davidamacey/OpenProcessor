"""Pins ``VisualSearchService.ingest_batch`` across its branches: the response, the
documents handed to OpenSearch, which Triton/face/OCR calls each image makes, and the
failure paths.

The golden file was captured from the implementation before ``ingest_batch`` was
decomposed. Triton, the face client, the OCR service and OpenSearch are scripted fakes;
hashes are the real imohash of the scripted bytes.
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


GOLDEN = Path(__file__).parent / 'golden' / 'ingest_batch_golden.json'


def _hash(data: bytes) -> str:
    return imohash.hashfileobject(io.BytesIO(data)).hex()


def _yolo(num: int = 0, *, shape: Any = (480, 640)) -> dict[str, Any]:
    out: dict[str, Any] = {
        'num_dets': num,
        'boxes': [[0, 0, 1, 1]] * num,
        'scores': [0.9 - 0.1 * i for i in range(num)],
        'classes': [[2, 0, 16][i % 3] for i in range(num)],
        'normalized_boxes': [[0.1, 0.1, 0.5, 0.5]] * num,
        'box_embeddings': [[0.5, 0.5]] * num,
        'image_embedding': [1.0, 0.0, 0.0],
    }
    if shape is not None:
        out['orig_shape'] = shape
    return out


def _faces(n: int, *, numpy_boxes: bool = False, short: bool = False) -> dict[str, Any]:
    boxes = [np.array([0.1, 0.1, 0.2, 0.2]) if numpy_boxes else [0.1, 0.1, 0.2, 0.2]] * n
    return {
        'num_faces': n,
        'face_embeddings': [[0.1, 0.2]] * n,
        'face_boxes': boxes,
        'face_scores': [0.99] * (n - 1 if short else n),
        'face_quality': [0.8] * (n - 1 if short else n),
        'face_landmarks': [[1.0] * 10] * n,
    }


def _ocr(n: int = 1, *, numpy: bool = False) -> dict[str, Any]:
    box = np.array([[1, 2, 3, 4]]) if numpy else [[1, 2, 3, 4]]
    return {
        'status': 'success',
        'num_texts': n,
        'texts': ['AB', 'CD'][:n],
        'boxes': box,
        'boxes_normalized': box,
        'det_scores': np.array([0.9]) if numpy else [0.9],
        'rec_scores': [0.8],
    }


IMG = {k: f'jpeg-bytes-{k}'.encode() for k in 'abcdef'}
IMAGES = [(IMG[k], f'id-{k}', None if k == 'a' else f'/p/{k}.jpg') for k in 'abcd']

# Per-image scripted pipeline outputs, keyed by the image letter.
BASE_SCRIPT: dict[str, dict[str, Any]] = {
    'a': {'yolo': _yolo(3), 'face': _faces(2), 'ocr': _ocr(2)},
    'b': {'yolo': _yolo(0, shape=None), 'face': _faces(0), 'ocr': {'status': 'failed'}},
    'c': {
        'yolo': _yolo(1),
        'face': _faces(3, numpy_boxes=True, short=True),
        'ocr': _ocr(1, numpy=True),
    },
    'd': {'yolo': _yolo(2), 'face': _faces(0), 'ocr': {'status': 'success', 'num_texts': 0}},
}

SCENARIOS: dict[str, dict[str, Any]] = {
    'empty': {'images': []},
    'full': {},
    'no_skip_duplicates': {'kwargs': {'skip_duplicates': False}},
    'some_duplicates': {'existing': ['b', 'd']},
    'all_duplicates': {'existing': ['a', 'b', 'c', 'd']},
    'only_detection': {
        'kwargs': {'enable_clip': False, 'enable_faces': False, 'enable_ocr': False},
    },
    'only_clip': {
        'kwargs': {'enable_detection': False, 'enable_faces': False, 'enable_ocr': False},
    },
    'only_faces': {
        'kwargs': {'enable_detection': False, 'enable_clip': False, 'enable_ocr': False},
    },
    'only_ocr': {
        'kwargs': {'enable_detection': False, 'enable_clip': False, 'enable_faces': False},
    },
    'nothing_enabled': {
        'kwargs': {
            'enable_detection': False,
            'enable_clip': False,
            'enable_faces': False,
            'enable_ocr': False,
        },
    },
    'inference_error_on_one': {'raise_for': ['c']},
    'inference_error_on_all': {'raise_for': ['a', 'b', 'c', 'd']},
    'jpeg_header_fallback': {'jpeg_dims': 'ok'},
    'jpeg_header_unreadable': {'jpeg_dims': 'fail'},
    'defer_heavy_ops': {'kwargs': {'defer_heavy_ops': True}},
    'near_duplicates_disabled': {'kwargs': {'detect_near_duplicates': False}},
    'near_duplicate_group_and_error': {'dups': {'a': 'joined', 'c': 'raise', 'd': 'created'}},
    'ocr_index_failure': {'ocr_index_fail': ['a']},
    'bulk_ingest_partial_result': {'bulk_result': {'global': 2}},
    'face_bulk_errors': {'face_bulk': {'indexed': 3, 'errors': ['x']}},
    'threshold_forwarded': {'kwargs': {'near_duplicate_threshold': 0.95}, 'dups': {'a': 'joined'}},
}


class FakeTriton:
    def __init__(self, script: dict[str, dict[str, Any]], raise_for: set[str]) -> None:
        self.script = {IMG[k]: v for k, v in script.items()}
        self.raise_for = {IMG[k] for k in raise_for}
        self.calls: list[str] = []

    def infer_yolo_clip_cpu(self, data: bytes) -> dict[str, Any]:
        self.calls.append(data.decode())
        if data in self.raise_for:
            msg = 'triton down'
            raise RuntimeError(msg)
        return self.script[data]['yolo']


class FakeFace:
    def __init__(self, script: dict[str, dict[str, Any]]) -> None:
        self.script = {IMG[k]: v for k, v in script.items()}
        self.calls: list[tuple[str, float]] = []

    def recognize(self, data: bytes, confidence: float) -> dict[str, Any]:
        self.calls.append((data.decode(), confidence))
        return self.script[data]['face']


class FakeOcr:
    def __init__(self, script: dict[str, dict[str, Any]]) -> None:
        self.script = {IMG[k]: v for k, v in script.items()}
        self.calls: list[tuple[str, bool]] = []

    def extract_text(self, data: bytes, filter_by_score: bool) -> dict[str, Any]:
        self.calls.append((data.decode(), filter_by_score))
        return self.script[data]['ocr']


class FakeOpenSearch:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.existing = {
            _hash(IMG[k]): {'image_id': f'old-{k}'} for k in scenario.get('existing', [])
        }
        self.bulk_result = scenario.get('bulk_result', {'global': 4, 'vehicles': 3, 'people': 1})
        self.face_bulk = scenario.get('face_bulk', {'indexed': 4})
        self.ocr_fail = {f'id-{k}' for k in scenario.get('ocr_index_fail', [])}
        self.calls: list[tuple[str, Any]] = []

    async def check_duplicates_by_hash_batch(self, hashes: list[str]) -> dict[str, Any]:
        self.calls.append(('check_duplicates_by_hash_batch', hashes))
        return {h: self.existing.get(h) for h in hashes}

    async def bulk_ingest(self, documents: list[dict[str, Any]], refresh: bool = False) -> Any:
        self.calls.append(('bulk_ingest', {'documents': documents, 'refresh': refresh}))
        return self.bulk_result

    async def bulk_index_faces(self, face_documents: list[dict[str, Any]]) -> Any:
        self.calls.append(('bulk_index_faces', face_documents))
        return self.face_bulk

    async def index_ocr_results(self, **kw: Any) -> None:
        self.calls.append(('index_ocr_results', kw))
        if kw['image_id'] in self.ocr_fail:
            msg = 'ocr index down'
            raise RuntimeError(msg)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {'ndarray': value.tolist()}
    if isinstance(value, dict):
        return {k: ('<ts>' if k == 'indexed_at' else _plain(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


async def run_scenario(name: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sc = SCENARIOS[name]
    images = sc.get('images', IMAGES)
    triton = FakeTriton(BASE_SCRIPT, set(sc.get('raise_for', [])))
    face = FakeFace(BASE_SCRIPT)
    ocr = FakeOcr(BASE_SCRIPT)
    opensearch = FakeOpenSearch(sc)
    monkeypatch.setattr('src.clients.triton_client.get_sync_triton_client', lambda url: triton)  # noqa: ARG005
    monkeypatch.setattr('src.clients.fast_face_client.get_fast_face_client', lambda url: face)  # noqa: ARG005
    monkeypatch.setattr('src.services.ocr_service.get_ocr_service', lambda: ocr)
    if 'jpeg_dims' in sc:

        def dims(data: bytes) -> tuple[int, int]:
            if sc['jpeg_dims'] == 'fail':
                msg = 'bad header'
                raise ValueError(msg)
            return (321, 123)

        monkeypatch.setattr('src.utils.affine.get_jpeg_dimensions_fast', dims)

    svc = VisualSearchService.__new__(VisualSearchService)
    svc.opensearch = opensearch  # type: ignore[assignment]
    svc.inference = None  # type: ignore[assignment]
    dup_calls: list[dict[str, Any]] = []

    async def assign(image_id: str, embedding: Any, threshold: float = 0.99) -> Any:
        dup_calls.append(
            {'image_id': image_id, 'embedding': _plain(embedding), 'threshold': threshold}
        )
        kind = sc.get('dups', {}).get(image_id.removeprefix('id-'))
        if kind == 'raise':
            msg = 'dup service down'
            raise RuntimeError(msg)
        if kind == 'joined':
            return {'action': 'joined_group', 'group_id': 'g1', 'similarity': 0.99}
        if kind == 'created':
            return {'action': 'created_group', 'group_id': 'g2', 'similarity': 0.97}
        return None

    svc._assign_to_duplicate_group = assign  # type: ignore[method-assign]
    result = await svc.ingest_batch(images, **sc.get('kwargs', {}))
    return _plain(
        {
            'result': result,
            'opensearch': opensearch.calls,
            'dup_calls': sorted(dup_calls, key=lambda c: c['image_id']),
            'triton': sorted(triton.calls),
            'face': sorted(face.calls),
            'ocr': sorted(ocr.calls),
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('name', sorted(SCENARIOS))
async def test_ingest_batch_matches_golden(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    got = await run_scenario(name, monkeypatch)
    # OCR indexing runs concurrently: compare it as a set, everything else in order.
    expected = json.loads(GOLDEN.read_text())[name]
    for snap in (got, expected):
        snap['opensearch'] = sorted(snap['opensearch'], key=lambda c: json.dumps(c, sort_keys=True))
    assert got == expected


def test_golden_file_has_exactly_the_scenarios() -> None:
    assert set(json.loads(GOLDEN.read_text())) == set(SCENARIOS)

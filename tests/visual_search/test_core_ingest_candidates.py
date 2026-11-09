"""The core ingest response lists the near-duplicate candidates: the stored images above the core
cut, by id and cosine. The group itself appears once ``near_dup_core_group`` has run."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core.dependencies import get_visual_search_service
from src.services.visual_search import VisualSearchService
from visual_search.core_fakes import INDEX, FakeCoreOpenSearch


NEW = np.array([1.0, 0.0, 0.0], dtype=np.float32)
CLOSE = np.array([0.9999, 0.0141, 0.0], dtype=np.float32)  # cosine 0.9999
NEAR = np.array([0.99, 0.141, 0.0], dtype=np.float32)  # cosine 0.99, still above the 0.98 cut
FAR = np.array([0.9, 0.436, 0.0], dtype=np.float32)  # cosine 0.9, below it


def _unit(v: np.ndarray) -> list[float]:
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def opensearch(monkeypatch: pytest.MonkeyPatch) -> FakeCoreOpenSearch:
    fake = FakeCoreOpenSearch()
    for name, vector in (('close', CLOSE), ('near', NEAR), ('far', FAR)):
        fake.docs[INDEX][name] = {'image_id': name, 'global_embedding': _unit(vector)}

    def infer(images: list[bytes], **_kw: Any) -> list[dict[str, Any]]:
        return [{'num_dets': 0, 'num_faces': 0, 'global_embedding': _unit(NEW)} for _ in images]

    monkeypatch.setattr('src.services.visual_search.ingest_batch.infer_batch', infer)
    monkeypatch.setattr('src.clients.triton_client.get_sync_triton_client', lambda _url: _Triton())
    return fake


class _Triton:
    def infer_yolo_clip_cpu(self, _data: bytes) -> dict[str, Any]:
        return {'num_dets': 0, 'image_embedding': _unit(NEW), 'orig_shape': (10, 10)}


def _service(fake: FakeCoreOpenSearch) -> VisualSearchService:
    service = VisualSearchService.__new__(VisualSearchService)
    service.opensearch = fake  # type: ignore[assignment]
    service.inference = None  # type: ignore[assignment]
    return service


@pytest.mark.asyncio
async def test_a_batch_lists_the_stored_images_above_the_core_cut_closest_first(
    opensearch,
) -> None:
    result = await _service(opensearch).ingest_batch(
        [(b'x', 'new', None)],
        enable_ocr=False,
        enable_detection=False,
        enable_faces=False,
    )

    (entry,) = result['near_duplicate_candidates']
    assert entry['image_id'] == 'new'
    assert [(c['image_id'], round(c['cosine'], 4)) for c in entry['candidates']] == [
        ('close', 0.9999),
        ('near', 0.99),
    ]
    assert result['near_duplicates'] == 1


@pytest.mark.asyncio
async def test_a_deferred_or_disabled_lookup_asks_the_index_nothing(opensearch) -> None:
    asked: list[Any] = []
    real = opensearch.msearch

    async def spy(*, body: Any) -> Any:
        asked.append(body)
        return await real(body=body)

    opensearch.msearch = spy  # type: ignore[method-assign]
    service = _service(opensearch)
    for flags in ({'defer_heavy_ops': True}, {'detect_near_duplicates': False}):
        result = await service.ingest_batch(
            [(f'x{flags}'.encode(), f'new-{len(asked)}', None)],
            enable_ocr=False,
            enable_detection=False,
            enable_faces=False,
            **flags,
        )
        assert result['near_duplicate_candidates'] is None
        assert result['near_duplicates'] == 0
    assert asked == []


@pytest.mark.asyncio
async def test_a_single_ingest_lists_its_candidates(opensearch) -> None:
    result = await _service(opensearch).ingest_image(
        b'x', 'new', enable_ocr=False, enable_detection=False, enable_faces=False
    )

    assert [
        (c['image_id'], round(c['cosine'], 4)) for c in result['near_duplicate_candidates']
    ] == [
        ('close', 0.9999),
        ('near', 0.99),
    ]


@pytest.mark.asyncio
async def test_a_single_ingest_with_nothing_above_the_cut_has_no_candidates(opensearch) -> None:
    del opensearch.docs[INDEX]['close'], opensearch.docs[INDEX]['near']

    result = await _service(opensearch).ingest_image(
        b'x', 'new', enable_ocr=False, enable_detection=False, enable_faces=False
    )

    assert 'near_duplicate_candidates' not in result


class _CandidateService:
    async def ingest_image(self, **_kw: object) -> dict[str, object]:
        return {
            'status': 'success',
            'image_id': 'i1',
            'near_duplicate_candidates': [{'image_id': 'close', 'cosine': 0.9999}],
        }

    async def ingest_batch(self, images_data: list[object], **_kw: object) -> dict[str, object]:
        return {
            'total': len(images_data),
            'processed': len(images_data),
            'near_duplicates': 1,
            'near_duplicate_candidates': [
                {'image_id': 'i1', 'candidates': [{'image_id': 'close', 'cosine': 0.9999}]}
            ],
        }


@pytest.fixture
def client():
    from src.main import app

    app.dependency_overrides[get_visual_search_service] = _CandidateService
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.pop(get_visual_search_service, None)


def test_the_routes_put_the_candidates_on_the_wire(client: TestClient) -> None:
    single = client.post('/ingest', files={'image': ('a.jpg', b'x', 'image/jpeg')})
    batch = client.post('/ingest/batch', files=[('images', ('a.jpg', b'x', 'image/jpeg'))])

    assert single.status_code == 201
    assert single.json()['near_duplicate_candidates'] == [{'image_id': 'close', 'cosine': 0.9999}]
    assert 'near_duplicate' not in single.json()
    assert batch.json()['near_duplicates'] == 1
    assert batch.json()['near_duplicate_candidates'] == [
        {'image_id': 'i1', 'candidates': [{'image_id': 'close', 'cosine': 0.9999}]}
    ]
    assert 'near_duplicate_details' not in batch.json()


def test_the_threshold_form_field_is_gone(client: TestClient) -> None:
    from src.main import app

    for path in ('/ingest', '/ingest/batch', '/ingest/directory'):
        params = app.openapi()['paths'][path]['post']
        body = params.get('requestBody', {}).get('content', {})
        form = next(iter(body.values()), {}).get('schema', {})
        names = set(form.get('properties', {})) | {p['name'] for p in params.get('parameters', [])}
        assert 'near_duplicate_threshold' not in names

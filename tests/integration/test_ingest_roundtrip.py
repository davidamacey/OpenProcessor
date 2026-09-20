"""Ingest -> crops/status round trip through the real app.

Ingests two images through ``POST /curation/ingest/batch`` against a
fake OpenSearch + a scripted detector/PE-encoder, then asserts
``GET /curation/crops`` lists both items with the quality fields
(``crop_area_norm``, ``blur_lap_var``, ``blur_lap_ratio``) populated,
and ``GET /curation/ingest/status`` reflects the new images.

Per plan §6.0 house rule, this fakes the OpenSearch/Triton I/O boundary
rather than standing up a live stack — see
``tests/integration/test_ingest_occ.py`` for the same convention.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


pytestmark = pytest.mark.integration


def _jpeg_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode='RGB').save(buf, format='JPEG', quality=95)
    return buf.getvalue()


class _FakeInferResult:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self._outputs[name]


class _FakeTritonPool:
    """One detection per image, normalized box [0.1,0.1,0.5,0.5], class 0."""

    async def infer(self, model_name: str, inputs: list, outputs: list) -> _FakeInferResult:  # noqa: ARG002
        boxes = np.array([[[0.1, 0.1, 0.5, 0.5]]], dtype=np.float32)
        return _FakeInferResult(
            {
                'num_dets': np.array([[1]], dtype=np.int32),
                'det_boxes': boxes,
                'det_scores': np.array([[0.95]], dtype=np.float32),
                'det_classes': np.array([[0]], dtype=np.float32),
            }
        )


class _FakePEEncoder:
    text_ready = False

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(crops), 1))

    async def embed_whole_frame(self, path: str) -> np.ndarray | None:  # noqa: ARG002
        return np.array([0.0, 1.0], dtype=np.float32)


class _FakeClassEntry:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.deprecated = False


class _FakeRegistryFile:
    def __init__(self) -> None:
        self.classes = [_FakeClassEntry(0, 'widget')]


class _FakeRegistry:
    def load(self) -> _FakeRegistryFile:
        return _FakeRegistryFile()

    def get(self, class_id: int) -> _FakeClassEntry | None:
        for c in self.load().classes:
            if c.class_id == class_id:
                return c
        return None


class _FakeOpenSearch:
    """Just enough of AsyncOpenSearch for ingest + /crops + /ingest/status."""

    def __init__(self) -> None:
        self.images: dict[str, dict[str, Any]] = {}
        self.items: dict[str, dict[str, Any]] = {}
        self.indices = self._Indices()

    class _Indices:
        async def exists(self, index: str) -> bool:  # noqa: ARG002
            return True

        async def create(self, index: str, body: dict) -> dict:  # noqa: ARG002
            return {'acknowledged': True}

        async def refresh(self, index: str) -> dict:  # noqa: ARG002
            return {'_shards': {}}

    def _items_index(self) -> str:
        from src.config import get_curation_config

        return get_curation_config().items_index

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        term = ((body.get('query') or {}).get('term') or {}).get('imohash')
        if term is not None:
            store = self.images
            hits = [
                {'_id': d.get('image_id', k), '_source': d}
                for k, d in store.items()
                if d.get('imohash') == term
            ]
            return {'hits': {'hits': hits[:1]}}
        # /curation/crops style query -- return every non-holdout item.
        if index == self._items_index():
            hits = [{'_id': k, '_source': d} for k, d in self.items.items()]
            size = body.get('size', len(hits))
            return {
                'hits': {'hits': hits[:size], 'total': {'value': len(hits)}},
            }
        # /curation/ingest/status style aggregation query over images.
        return {'hits': {'hits': [], 'total': {'value': len(self.images)}}, 'aggregations': {}}

    async def msearch(self, *, body: list[dict[str, Any]]) -> dict[str, Any]:
        responses = []
        for line in body[1::2]:
            term = ((line.get('query') or {}).get('term') or {}).get('imohash')
            hits = [
                {'_id': d.get('image_id', k), '_source': d}
                for k, d in self.images.items()
                if d.get('imohash') == term
            ]
            responses.append({'hits': {'hits': hits[:1]}})
        return {'responses': responses}

    async def mget(self, *, body: dict[str, Any], index: str) -> dict[str, Any]:  # noqa: ARG002
        docs = []
        for doc_id in body['ids']:
            if doc_id in self.items:
                docs.append(
                    {
                        '_id': doc_id,
                        'found': True,
                        '_source': self.items[doc_id],
                        '_seq_no': 1,
                        '_primary_term': 1,
                    }
                )
            else:
                docs.append({'_id': doc_id, 'found': False})
        return {'docs': docs}

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        items_index = self._items_index()
        result_items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            if 'index' in action:
                meta = action['index']
                store = self.items if meta['_index'] == items_index else self.images
                store[meta['_id']] = doc
                result_items.append({'index': {'_id': meta['_id'], 'status': 201}})
            elif 'create' in action:
                meta = action['create']
                if meta['_id'] in self.items:
                    result_items.append({'create': {'_id': meta['_id'], 'status': 409}})
                else:
                    self.items[meta['_id']] = doc
                    result_items.append({'create': {'_id': meta['_id'], 'status': 201}})
        return {'errors': False, 'items': result_items}

    async def update(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        body: dict[str, Any],
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.items.setdefault(id, {}).update(body['doc'])
        return {'_id': id, 'result': 'updated', '_seq_no': 2, '_primary_term': 1}

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        return {'_id': id, '_source': self.items.get(id, {}), '_seq_no': 1, '_primary_term': 1}

    async def count(self, index: str, body: dict | None = None) -> dict[str, Any]:  # noqa: ARG002
        return {'count': len(self.items)}


@pytest.fixture
def fake_opensearch() -> _FakeOpenSearch:
    return _FakeOpenSearch()


@pytest.fixture
def client(fake_opensearch: _FakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import src.main as main_module
    from src.core.dependencies import get_async_triton, get_opensearch
    from src.routers.curation._common import _raw_opensearch_dep, _registry_dep

    main_module.app.dependency_overrides[get_opensearch] = lambda: fake_opensearch
    main_module.app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch
    main_module.app.dependency_overrides[get_async_triton] = lambda: _FakeTritonPool()
    main_module.app.dependency_overrides[_registry_dep] = lambda: _FakeRegistry()

    monkeypatch.setenv('OP_DETECTION_DETECTOR_MODEL', 'fake_item_detector')
    # ingest.py's service factory reaches AppResources.async_triton_pool /
    # app.state.pe_encoder directly (not via FastAPI Depends()), and the
    # lifespan startup below unconditionally (re)builds both -- so these
    # must be patched AFTER entering the TestClient context, not before.
    monkeypatch.setattr(main_module, 'get_async_triton_pool', lambda: _FakeTritonPool())

    with TestClient(main_module.app) as c:
        main_module.app.state.pe_encoder = _FakePEEncoder()
        yield c

    main_module.app.dependency_overrides.clear()


def test_ingest_batch_then_crops_and_status(
    client: TestClient, fake_opensearch: _FakeOpenSearch
) -> None:
    body = {
        'items': [
            {'path': '/tmp/roundtrip_a.jpg', 'source': 'roundtrip_test'},
            {'path': '/tmp/roundtrip_b.jpg', 'source': 'roundtrip_test'},
        ]
    }
    # Write real files so the router's Path(...).read_bytes() succeeds.
    from pathlib import Path

    Path('/tmp/roundtrip_a.jpg').write_bytes(_jpeg_bytes(1))
    Path('/tmp/roundtrip_b.jpg').write_bytes(_jpeg_bytes(2))

    resp = client.post('/curation/ingest/batch', json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload['summary']['successful'] == 2
    assert payload['summary']['crops_indexed'] == 2

    crops_resp = client.get('/curation/crops')
    assert crops_resp.status_code == 200, crops_resp.text
    crops_payload = crops_resp.json()
    items = crops_payload.get('items') or crops_payload.get('crops') or crops_payload
    assert isinstance(items, list)
    assert len(items) == 2
    for item in items:
        # crops.py's wire model surfaces crop_area_norm/crop_rank_in_image/
        # blur_lap_ratio (not blur_lap_var, which is diagnostic-only and
        # excluded from that response by design) -- checked against the
        # response here, and against the raw OpenSearch doc below.
        assert item.get('crop_area_norm') is not None
        assert item.get('blur_lap_ratio') is not None
        assert item.get('crop_rank_in_image') is not None

    assert len(fake_opensearch.items) == 2
    for doc in fake_opensearch.items.values():
        assert doc.get('blur_lap_var') is not None
        assert doc.get('blur_lap_ratio') is not None
        assert doc.get('crop_area_norm') is not None
        assert doc.get('pe_embedding') is not None

    status_resp = client.get('/curation/ingest/status')
    assert status_resp.status_code == 200, status_resp.text
    status_payload = status_resp.json()
    assert status_payload['total'] == 2

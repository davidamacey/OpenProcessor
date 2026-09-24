"""Tests for GET /curation/search/text.

Mounts the real curation router with OpenSearch + the PE text encoder
both stubbed — no real torch/perception_models involved, matching this
repo's existing convention (test_methods_router.py, test_pe_encoder.py)
of keeping router-level tests independent of the heavy ML stack.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


PE_DIM = 1024


def _fake_pe_encoder(ready: bool = True) -> MagicMock:
    enc = MagicMock()
    enc.text_ready = ready
    vec = np.zeros((1, PE_DIM), dtype=np.float32)
    vec[0, 0] = 1.0
    enc.encode_text = MagicMock(return_value=vec)
    return enc


def _fake_opensearch(hits: list[dict] | None = None) -> AsyncMock:
    fake_os = AsyncMock()
    fake_os.search = AsyncMock(
        return_value={'hits': {'hits': hits or [], 'total': {'value': len(hits or [])}}}
    )
    fake_os.count = AsyncMock(return_value={'count': 0})
    return fake_os


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch):
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setenv('OP_SEMANTIC_SEARCH_ENABLED', 'true')

    fake_os = _fake_opensearch()
    fake_encoder = _fake_pe_encoder()

    app = FastAPI()
    app.include_router(curation_router)
    app.state.pe_encoder = fake_encoder
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    client.fake_encoder = fake_encoder  # type: ignore[attr-defined]
    return client


def test_search_text_disabled_returns_400(monkeypatch: pytest.MonkeyPatch, app_client: TestClient):
    monkeypatch.setenv('OP_SEMANTIC_SEARCH_ENABLED', 'false')
    resp = app_client.get('/curation/search/text', params={'q': 'white pickup truck'})
    assert resp.status_code == 400
    assert 'OP_SEMANTIC_SEARCH_ENABLED' in resp.json()['detail']


def test_search_text_encoder_not_ready_returns_503(app_client: TestClient):
    app_client.fake_encoder.text_ready = False
    resp = app_client.get('/curation/search/text', params={'q': 'white pickup truck'})
    assert resp.status_code == 503
    assert 'pe_text' in resp.json()['detail']


def test_search_text_happy_path(app_client: TestClient):
    hit = {
        '_id': 'crop_1',
        '_score': 0.83,
        '_source': {
            'crop_id': 'crop_1',
            'image_path': '/data/foo.jpg',
            'class_id': 3,
            'class_name': 'pickup',
        },
    }
    app_client.fake_os.search = AsyncMock(
        return_value={'hits': {'hits': [hit], 'total': {'value': 1}}}
    )
    resp = app_client.get(
        '/curation/search/text', params={'q': 'white pickup truck', 'page': 1, 'page_size': 30}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body['total'] == 1
    assert body['page'] == 1
    assert body['page_size'] == 30
    assert len(body['items']) == 1
    item = body['items'][0]
    assert item['crop_id'] == 'crop_1'
    assert item['class_name'] == 'pickup'
    assert item['semantic_score'] == 0.83

    # encode_text was offloaded, never called with the wrong signature.
    app_client.fake_encoder.encode_text.assert_called_once_with(['white pickup truck'])

    # kNN query body never carries pe_embedding/v6_embedding/region_embedding
    # or class_id_history (F-25 -- this is a paginated list endpoint).
    _args, kwargs = app_client.fake_os.search.call_args
    body_sent = kwargs.get('body') or _args[-1]
    assert set(body_sent['_source']['excludes']) == {
        'pe_embedding',
        'v6_embedding',
        'region_embedding',
        'class_id_history',
    }
    assert 'knn' in body_sent['query']
    assert 'pe_embedding' in body_sent['query']['knn']


def test_search_text_min_score_filters_results(app_client: TestClient):
    """F-24: min_score is applied by OpenSearch itself now (top-level
    request body field), not filtered out of the full hit list in
    Python -- so the fake must apply it like real OpenSearch would."""
    hits = [
        {'_id': 'a', '_score': 0.9, '_source': {'crop_id': 'a'}},
        {'_id': 'b', '_score': 0.1, '_source': {'crop_id': 'b'}},
    ]

    async def _search(*, index: str, body: dict) -> dict:
        pool = hits
        min_score = body.get('min_score')
        if min_score is not None:
            pool = [h for h in pool if h['_score'] >= min_score]
        frm = body.get('from', 0)
        size = body.get('size', len(pool))
        return {'hits': {'hits': pool[frm : frm + size], 'total': {'value': len(pool)}}}

    app_client.fake_os.search = AsyncMock(side_effect=_search)
    resp = app_client.get('/curation/search/text', params={'q': 'red sedan', 'min_score': 0.5})
    assert resp.status_code == 200
    body = resp.json()
    assert body['total'] == 1
    assert body['items'][0]['crop_id'] == 'a'


def test_search_text_empty_query_string_rejected(app_client: TestClient):
    resp = app_client.get('/curation/search/text', params={'q': ''})
    assert resp.status_code == 422


def test_search_text_no_hits_returns_empty(app_client: TestClient):
    app_client.fake_os.search = AsyncMock(
        return_value={'hits': {'hits': [], 'total': {'value': 0}}}
    )
    resp = app_client.get('/curation/search/text', params={'q': 'nonexistent thing'})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {'items': [], 'total': 0, 'page': 1, 'page_size': 30}

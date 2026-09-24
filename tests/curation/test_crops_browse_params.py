"""``GET /crops`` honors ``limit``, ``sort``, ``conf_min``/``conf_max`` and,
for ``order=diverse``, ``k`` (contract audit S10)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import _common, select


class _RecordingOS:
    def __init__(self, n_docs: int = 3) -> None:
        self.docs = {f'c{i}': {'crop_id': f'c{i}', 'confidence': 0.5} for i in range(n_docs)}
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        json.dumps(body)  # a FieldInfo leaking into the query would blow up here
        self.bodies.append(body)
        hits = [{'_id': k, '_source': v} for k, v in self.docs.items()]
        return {'hits': {'total': {'value': len(hits)}, 'hits': hits[: body.get('size', 10)]}}

    async def mget(self, *, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:  # noqa: ARG002
        return {
            'docs': [
                {'_id': i, '_source': self.docs[i], 'found': True}
                for i in body['ids']
                if i in self.docs
            ]
        }

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {'count': len(self.docs)}


@pytest.fixture
def fake_os() -> _RecordingOS:
    return _RecordingOS()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake_os: _RecordingOS) -> Any:
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', True)
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


P = _common.config.api_prefix


def test_limit_sets_the_page_size(client: TestClient, fake_os: _RecordingOS) -> None:
    r = client.get(f'{P}/crops', params={'limit': 2, 'page_size': 50})
    assert r.status_code == 200, r.text
    assert fake_os.bodies[-1]['size'] == 2
    assert r.json()['page_size'] == 2
    assert len(r.json()['crops']) == 2


def test_limit_is_bounded(client: TestClient) -> None:
    assert client.get(f'{P}/crops', params={'limit': 0}).status_code == 422
    assert client.get(f'{P}/crops', params={'limit': 501}).status_code == 422


def test_default_sort_is_newest_first(client: TestClient, fake_os: _RecordingOS) -> None:
    assert client.get(f'{P}/crops').status_code == 200
    sort = fake_os.bodies[-1]['sort']
    assert sort[0]['updated_at']['order'] == 'desc'
    # F-7: stable crop_id tiebreaker, always last.
    assert sort[-1] == {'crop_id': {'order': 'asc'}}


def test_sort_param_is_applied(client: TestClient, fake_os: _RecordingOS) -> None:
    r = client.get(f'{P}/crops', params={'sort': 'confidence:asc'})
    assert r.status_code == 200, r.text
    sort = fake_os.bodies[-1]['sort']
    assert sort[0]['confidence']['order'] == 'asc'
    assert sort[-1] == {'crop_id': {'order': 'asc'}}
    r = client.get(f'{P}/crops', params={'sort': 'updated_at:desc'})
    assert r.status_code == 200
    assert fake_os.bodies[-1]['sort'][0]['updated_at']['order'] == 'desc'


@pytest.mark.parametrize('bad', ['nope:asc', 'confidence:sideways', 'pe_embedding'])
def test_unknown_sort_is_400(client: TestClient, bad: str) -> None:
    r = client.get(f'{P}/crops', params={'sort': bad})
    assert r.status_code == 400
    assert 'sort' in r.json()['detail']


def test_confidence_band(client: TestClient, fake_os: _RecordingOS) -> None:
    r = client.get(f'{P}/crops', params={'conf_min': 0.2, 'conf_max': 0.6})
    assert r.status_code == 200, r.text
    filt = fake_os.bodies[-1]['query']['bool']['filter']
    assert {'range': {'confidence': {'gte': 0.2, 'lte': 0.6}}} in filt
    assert client.get(f'{P}/crops', params={'conf_min': 0.7, 'conf_max': 0.1}).status_code == 400


def test_class_crops_passes_real_defaults(client: TestClient, fake_os: _RecordingOS) -> None:
    """/classes/{id}/crops calls list_crops directly; unset params must be
    plain defaults, not FastAPI FieldInfo objects leaking into the query.

    F-19: class_id/test_holdout/class_excluded are pure predicates and
    live in filter context now, but the optional params this test cares
    about (max_rank, min_blur_ratio, classifier_conf_lt, item_text,
    confidence band) must still be absent when unset."""
    r = client.get(f'{P}/classes/3/crops')
    assert r.status_code == 200, r.text
    body = fake_os.bodies[-1]
    assert body['size'] == 50
    filt = body['query']['bool']['filter']
    assert not any('range' in clause for clause in filt)
    assert not any(
        'should' in clause.get('bool', {}) for clause in filt if isinstance(clause, dict)
    )


def test_diverse_order_honors_k(
    monkeypatch: pytest.MonkeyPatch, fake_os: _RecordingOS, client: TestClient
) -> None:
    fake_os.docs = {f'c{i}': {'crop_id': f'c{i}'} for i in range(6)}
    ids = list(fake_os.docs)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(ids), 8)).astype(np.float32)

    async def _fake_fetch(*_: Any, **__: Any) -> tuple[list[str], np.ndarray, bool]:
        return ids, emb, False

    monkeypatch.setenv('OP_SELECT_DIVERSE_ENABLED', '1')
    monkeypatch.setattr(select, 'fetch_pool_embeddings', _fake_fetch)
    select._ORDER_CACHE.clear()

    r = client.get(f'{P}/crops', params={'order': 'diverse', 'k': 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['method'] == 'diverse'
    assert body['total'] == 2
    assert len(body['crops']) == 2
    assert body['n_pool'] == 6

    r = client.get(f'{P}/crops', params={'order': 'diverse'})
    assert r.json()['total'] == 6


def test_page_too_deep_is_422(client: TestClient) -> None:
    """F-7: from+size past the 10000 result-window ceiling must 422
    explicitly rather than let OpenSearch 500 past index.max_result_window."""
    r = client.get(f'{P}/crops', params={'page': 400, 'page_size': 30})
    assert r.status_code == 422, r.text


def test_page_within_window_is_fine(client: TestClient) -> None:
    r = client.get(f'{P}/crops', params={'page': 300, 'page_size': 30})
    assert r.status_code == 200, r.text

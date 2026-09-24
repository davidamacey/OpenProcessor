"""F-25: list endpoints stop shipping ``class_id_history`` in their
``_source`` fetch. Covers the query-body shape for every list endpoint
that switched to ``item_list_source_excludes`` (crops, review, regions,
region training candidates, semantic search) and confirms
``label_undo.py``'s internal history fetch is untouched (it needs the
field, unlike every list renderer).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import _common


class _RecordingOS:
    def __init__(self, n_docs: int = 2) -> None:
        self.docs = {
            f'c{i}': {'crop_id': f'c{i}', 'confidence': 0.5, 'class_id_history': [{'x': 1}]}
            for i in range(n_docs)
        }
        self.search_bodies: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        json.dumps(body)  # a FieldInfo leaking into the query would blow up here
        self.search_bodies.append(body)
        hits = [{'_id': k, '_source': v} for k, v in self.docs.items()]
        return {'hits': {'total': {'value': len(hits)}, 'hits': hits[: body.get('size', 10)]}}

    async def get(self, *, index: str, id: str, **kw: Any) -> dict[str, Any]:  # noqa: A002
        self.get_calls.append({'index': index, 'id': id, **kw})
        doc = self.docs.get(id, {'crop_id': id})
        return {'_id': id, '_source': dict(doc), 'found': True}

    async def mget(self, *, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:  # noqa: ARG002
        return {
            'docs': [
                {'_id': i, '_source': self.docs[i], 'found': True}
                for i in body['ids']
                if i in self.docs
            ]
        }


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


def test_crops_list_excludes_class_id_history(client: TestClient, fake_os: _RecordingOS) -> None:
    r = client.get(f'{P}/crops')
    assert r.status_code == 200, r.text
    excludes = fake_os.search_bodies[-1]['_source']['excludes']
    assert 'class_id_history' in excludes
    # Response shape: no crop in the list carries class_id_history at all
    # (serialize_item never emits it for any endpoint -- the exclude is a
    # network-cost fix, not a wire-shape change).
    assert all('class_id_history' not in c for c in r.json()['crops'])


def test_review_queue_excludes_class_id_history(client: TestClient, fake_os: _RecordingOS) -> None:
    r = client.get(f'{P}/review/all')
    assert r.status_code == 200, r.text
    excludes = fake_os.search_bodies[-1]['_source']['excludes']
    assert 'class_id_history' in excludes
    assert all('class_id_history' not in item for item in r.json()['items'])


def test_regions_list_excludes_class_id_history(client: TestClient, fake_os: _RecordingOS) -> None:
    r = client.get(f'{P}/regions')
    assert r.status_code == 200, r.text
    excludes = fake_os.search_bodies[-1]['_source']['excludes']
    assert 'class_id_history' in excludes


def test_crop_single_get_keeps_class_id_history_source_but_excludes_vectors(
    client: TestClient, fake_os: _RecordingOS
) -> None:
    """GET /crops/{id} is a single-item view -- it may legitimately want
    history, so it only drops the embedding vectors (matching every other
    endpoint's behavior), not class_id_history."""
    r = client.get(f'{P}/crops/c0')
    assert r.status_code == 200, r.text
    assert fake_os.get_calls
    excludes = fake_os.get_calls[-1]['_source_excludes']
    assert 'class_id_history' not in excludes
    assert 'pe_embedding' in excludes


def test_label_undo_internal_fetch_still_reads_class_id_history() -> None:
    """label_undo.py's own internal history fetch (distinct from any list
    endpoint) must keep reading class_id_history -- it's the one caller
    that actually needs it."""
    import inspect

    from src.routers.curation import label_undo

    source = inspect.getsource(label_undo)
    assert "_source_includes=['class_id_history']" in source


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

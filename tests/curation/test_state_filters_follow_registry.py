"""The embedding filters and the review ``all`` tab read the primary space, not the 0.6.0 names.

After a flip the vector is ``vec_<key>`` and its state ``vec_<key>_state``; a filter that still
asked for ``pe_embedding`` / ``embedding_state`` would count the retired space (embeddings plan
5.4).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from _embedding_registry_fake import new_space
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.embedding_spaces import Target
from src.routers.curation import _common
from src.services.curation import index_bootstrap


VECTOR = 'vec_mobileclip2_s2_v1'
STATE = 'vec_mobileclip2_s2_v1_state'


class _RecordingOS:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:  # noqa: ARG002
        self.bodies.append(body)
        return {'hits': {'total': {'value': 0}, 'hits': []}, 'aggregations': {}}

    async def count(self, *, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:  # noqa: ARG002
        self.bodies.append(body)
        return {'count': 0}


@pytest.fixture
def fake_os() -> _RecordingOS:
    return _RecordingOS()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake_os: _RecordingOS, embedding_registry: Any) -> Any:
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'default'})
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    embedding_registry.add(new_space(Target.ITEM, 'mobileclip2_s2'), primary=True)
    app = mount_curation_routers(FastAPI(), curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


P = f'{_common.config.api_prefix}/projects/default'


def _last_query(fake_os: _RecordingOS) -> str:
    """The query part of the last body (the ``_source`` excludes still list every live vector)."""
    return json.dumps(fake_os.bodies[-1]['query'])


def test_the_embedded_filter_asks_for_the_primary_vector(
    client: TestClient, fake_os: _RecordingOS
) -> None:
    r = client.get(f'{P}/crops', params={'embedding_state': 'embedded'})
    assert r.status_code == 200, r.text
    query = _last_query(fake_os)
    assert VECTOR in query
    assert 'pe_embedding' not in query


def test_a_no_vector_reason_filters_on_the_primary_state_field(
    client: TestClient, fake_os: _RecordingOS
) -> None:
    r = client.get(f'{P}/crops', params={'embedding_state': 'failed'})
    assert r.status_code == 200, r.text
    query = _last_query(fake_os)
    assert STATE in query
    assert '"embedding_state"' not in query


def test_the_review_all_tab_asks_for_the_primary_vector(
    client: TestClient, fake_os: _RecordingOS
) -> None:
    r = client.get(f'{P}/review/all')
    assert r.status_code == 200, r.text
    query = _last_query(fake_os)
    assert VECTOR in query
    assert 'pe_embedding' not in query


def test_semantic_search_requires_the_primary_vector(client: TestClient) -> None:
    r = client.get(f'{P}/methods')
    assert r.status_code == 200, r.text
    (entry,) = [s for s in r.json()['strategies'] if s['id'] == 'semantic_search']
    assert entry['requires_field'] == VECTOR

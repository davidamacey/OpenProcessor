"""The per-item decision routes (undo, discard, VLM dismiss, review un-dismiss, history) map
failures by type: only a missing document is a 404; an OpenSearch outage is the 503 outage
response, never "crop not found"."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opensearchpy.exceptions import ConnectionError as OsConnectionError

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config


ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default'


class _DownOpenSearch(QueryFakeOpenSearch):
    """Answers nothing: every document read fails like an unreachable cluster."""

    async def get(self, **_kw: Any) -> dict[str, Any]:
        raise OsConnectionError('N/A', 'connection refused', None)

    async def mget(self, **_kw: Any) -> dict[str, Any]:
        raise OsConnectionError('N/A', 'connection refused', None)


def _client(fake: Any) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


_WRITES = [
    ('post', '/crops/c1/label/undo', None),
    ('delete', '/crops/c1/label', None),
    ('post', '/crops/c1/discard', {'clear_class': True}),
    ('post', '/crops/c1/vlm_dismiss', None),
    ('post', '/crops/c1/review_undismiss', None),
]


@pytest.mark.parametrize(('method', 'path', 'body'), _WRITES)
def test_an_opensearch_outage_is_not_a_404(method: str, path: str, body: Any) -> None:
    fake = _DownOpenSearch({ITEMS: {'c1': {'crop_id': 'c1'}}})
    r = _client(fake).request(method, BASE + path, json=body)
    assert r.status_code == 503, r.text
    assert 'not found' not in r.text


def test_history_outage_is_a_503() -> None:
    fake = _DownOpenSearch({ITEMS: {'c1': {'crop_id': 'c1'}}})
    assert _client(fake).get(f'{BASE}/crops/c1/history').status_code == 503


@pytest.mark.parametrize(('method', 'path', 'body'), _WRITES)
def test_a_missing_item_is_a_404(method: str, path: str, body: Any) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {}})
    r = _client(fake).request(method, BASE + path, json=body)
    assert r.status_code == 404, r.text
    assert r.json()['detail'] == 'crop not found: c1'


def test_history_of_a_missing_item_is_a_404() -> None:
    fake = QueryFakeOpenSearch({ITEMS: {}})
    assert _client(fake).get(f'{BASE}/crops/c1/history').status_code == 404

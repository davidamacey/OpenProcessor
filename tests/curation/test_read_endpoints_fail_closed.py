"""Read endpoints fail closed: an OpenSearch outage is a 503, never an
empty/zero answer that looks like real data.

``GET /ingest/sam_drain`` used to answer ``total_unfinished: 0`` on an
outage — exactly the "worker has caught up" signal an ingest walker
waits for. Same shape of bug on ``/ingest/status``, ``/classes`` (counts)
and ``/stats/classes`` (registry join). Single-item reads tell a missing
item (404) from an outage (503).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _down() -> AsyncMock:
    fake = AsyncMock()
    boom = AsyncMock(side_effect=RuntimeError('connection refused'))
    fake.search = boom
    fake.count = boom
    fake.get = boom
    return fake


@pytest.mark.parametrize(
    'path',
    [
        '/curation/ingest/sam_drain',
        '/curation/ingest/status',
        '/curation/classes',
        '/curation/review/regions/locate?crop_id=c1',
        '/curation/crops/c1/history',
    ],
)
def test_outage_is_503(path: str) -> None:
    r = _client(_down()).get(path)
    assert r.status_code == 503, (path, r.status_code, r.text)


def test_stats_classes_registry_failure_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenRegistry:
        def load(self) -> Any:
            raise OSError('registry unreadable')

    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: _BrokenRegistry())
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'aggregations': {'by_class': {'buckets': []}}})
    r = _client(fake).get('/curation/stats/classes')
    assert r.status_code == 503, r.text


def test_missing_item_is_still_404() -> None:
    fake = AsyncMock()
    fake.get = AsyncMock(side_effect=KeyError('c1'))
    assert _client(fake).get('/curation/crops/c1/history').status_code == 404

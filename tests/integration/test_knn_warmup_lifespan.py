"""Proves the FastAPI startup lifespan fires the kNN warmup task.

Monkeypatches ``warm_knn_indexes`` before constructing the app so this
never issues a real ``_plugins/_knn/warmup`` call; it only proves the
lifespan wiring calls it. The function's own behavior (endpoint URL,
swallowing failures) is covered by ``tests/curation/test_knn_warmup.py``.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def _wait_until(predicate, timeout: float = 5.0, poll: float = 0.02) -> bool:
    """Poll ``predicate`` from the test thread while the app's loop (in
    TestClient's background thread) runs the fire-and-forget task."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


def test_lifespan_fires_knn_warmup_in_background(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stubs the OpenSearch client + index bootstrap so this exercises the
    lifespan wiring without depending on a reachable OpenSearch (the
    configured URL is the in-cluster ``opensearch:9200`` hostname, not
    reachable from this test host)."""
    calls: list[object] = []

    async def _fake_warm_knn_indexes(opensearch: object) -> None:
        calls.append(opensearch)

    async def _fake_create_curation_indexes(*_args, **_kwargs) -> None:
        return None

    fake_wrapper = AsyncMock()
    fake_wrapper.client = AsyncMock()

    monkeypatch.setattr('src.routers.curation._common.warm_knn_indexes', _fake_warm_knn_indexes)
    monkeypatch.setattr(
        'src.clients.curation_opensearch.create_curation_indexes',
        _fake_create_curation_indexes,
    )
    monkeypatch.setattr(
        'src.core.dependencies.OpenSearchClientFactory.get_client',
        AsyncMock(return_value=fake_wrapper),
    )

    from src.main import app

    with TestClient(app):
        assert _wait_until(lambda: bool(calls)), 'lifespan never scheduled warm_knn_indexes'

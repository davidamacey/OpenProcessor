"""List routes refuse a page past the OpenSearch result window with 422, not a 503."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()
    fake_os.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})
    fake_os.count = AsyncMock(return_value={'count': 0})
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


ROUTES = ['/audit/queue', '/classes/3/crops']


@pytest.mark.parametrize('route', ROUTES)
def test_a_page_past_the_result_window_is_422(client: TestClient, route: str) -> None:
    r = client.get(f'/curation/projects/default{route}', params={'page': 100000})
    assert r.status_code == 422, r.text


@pytest.mark.parametrize('route', ROUTES)
def test_page_one_is_200(client: TestClient, route: str) -> None:
    r = client.get(f'/curation/projects/default{route}', params={'page': 1})
    assert r.status_code == 200, r.text

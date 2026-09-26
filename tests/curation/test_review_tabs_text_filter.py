"""``GET /review/tabs``: the ``regions`` tab advertises a ``text`` filter
only when the active region profile reads text."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.review_queries import tab_filters


@pytest.fixture
def client() -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


@pytest.fixture
def profile_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.services.detection import profile_registry

    def _activate(text_reader: str) -> None:
        monkeypatch.setenv('OP_REGION_DETECTION_TEXT_READER', text_reader)
        profile_registry._reset_registry_for_tests()

    yield _activate
    profile_registry._reset_registry_for_tests()


def _regions_tab(client: TestClient) -> dict[str, Any]:
    resp = client.get('/curation/review/tabs')
    assert resp.status_code == 200, resp.text
    return next(t for t in resp.json()['tabs'] if t['id'] == 'regions')


def test_text_free_profile_has_no_text_filter(client: TestClient, profile_env: Any) -> None:
    profile_env('none')
    assert 'text' not in tab_filters('regions')
    tab = _regions_tab(client)
    assert 'text' not in tab['filters']
    assert 'region_status' in tab['filters']


def test_text_reading_profile_keeps_the_text_filter(client: TestClient, profile_env: Any) -> None:
    profile_env('vlm_then_ocr')
    assert 'text' in tab_filters('regions')
    assert 'text' in _regions_tab(client)['filters']

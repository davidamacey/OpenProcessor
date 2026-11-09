"""Shared fixtures for the region-profile router tests: the cache reset and the
app over an in-memory OpenSearch with a READY model repository.

Registered as a plugin in ``tests/conftest.py`` (pytest ``pytest_plugins``,
https://docs.pytest.org/en/stable/how-to/writing_plugins.html).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def region_profile_caches():
    from src.services.config_store.store import reset_config_stores
    from src.services.detection import profile_registry

    reset_config_stores()
    profile_registry._reset_registry_for_tests()
    yield
    reset_config_stores()
    profile_registry._reset_registry_for_tests()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from curation.query_fakes import QueryFakeOpenSearch
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = QueryFakeOpenSearch({})

    async def _fake_repo_index() -> list[dict]:
        return [
            {'name': 'my_detector', 'state': 'READY', 'version': '1'},
            {'name': 'paddleocr_det_trt', 'state': 'READY', 'version': '1'},
            {'name': 'paddleocr_rec_trt', 'state': 'READY', 'version': '1'},
            {'name': 'ocr_pipeline', 'state': 'READY', 'version': '1'},
        ]

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    from src.services.models.triton_control import TritonControlService

    monkeypatch.setattr(
        TritonControlService, 'get_repository_index', lambda _self: _fake_repo_index()
    )

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client

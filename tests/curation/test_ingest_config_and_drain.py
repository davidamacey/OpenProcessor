"""`GET /curation/ingest/config`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    import pytest


def _client(fake: AsyncMock) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_ingest_config_is_typed_and_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config.curation_accessors as curation_config_mod

    monkeypatch.setenv('OP_UPLOAD_MAX_IMAGES_PER_REQUEST', '7')
    monkeypatch.setenv('OP_UPLOAD_MAX_BYTES_PER_REQUEST', '1000')
    monkeypatch.setenv('OP_UPLOAD_ACCEPTED_EXTENSIONS', '.jpg,.png')
    monkeypatch.setenv('OP_BATCH_MAX_ITEMS_PER_REQUEST', '9')
    monkeypatch.setenv('OP_REGION_DRAIN_POLL_INTERVAL_S', '5')
    monkeypatch.setenv('OP_REGION_DRAIN_STABLE_POLLS', '2')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)

    fake = AsyncMock()
    fake.get.side_effect = RuntimeError('404 not_found')
    r = _client(fake).get('/curation/projects/default/ingest/config')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['policy']['embedding']['mode'] == 'all'
    assert body['upload']['max_images_per_request'] == 7
    assert body['upload']['max_bytes_per_request'] == 1000
    assert body['upload']['accepted_extensions'] == ['.jpg', '.png']
    assert body['upload']['persists_bytes'] is True
    assert body['batch']['max_items'] == 9
    assert isinstance(body['batch']['source_roots'], list)
    assert body['region_drain']['poll_interval_s'] == 5
    assert body['region_drain']['stable_polls'] == 2

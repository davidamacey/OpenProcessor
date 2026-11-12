"""Deleting a preset that an activation points at is 409 ``in_use``, like every other config kind."""

from __future__ import annotations

import pytest
from _curation_app import SCOPED, mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.db.repos.config import ConfigInUseError
from src.routers.curation import research_configs, router as curation_router


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    async def _refused(*_a: object, **_k: object) -> None:
        raise ConfigInUseError('strict')

    monkeypatch.setattr(research_configs, 'delete_document', _refused)
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize('path', ['cluster_profiles', 'assist_loop_presets', 'near_dup_profiles'])
def test_deleting_the_active_preset_is_409_in_use(client: TestClient, path: str) -> None:
    resp = client.delete(f'{SCOPED}/{path}/strict', params={'expected_revision': 1})
    assert resp.status_code == 409, resp.text
    assert resp.json()['detail']['error'] == 'in_use'

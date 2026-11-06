"""Routes and the repair scan that read the project's ingest profile see the
activation made through another worker, and refuse (503) rather than answer from
the deployment profile when the config store cannot be read."""

from __future__ import annotations

from typing import Any

import pytest
from _class_registry_fake import InMemoryRegistry
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._ingest_profile_world import Pool, WorldOpenSearch, body
from src.services.config_store.store import reset_config_stores
from src.services.curation import index_bootstrap


BASE = '/curation/projects/default'


class _ConfigsDown(WorldOpenSearch):
    """The config store's revision read fails; everything else works."""

    down = False

    async def get(self, index: str, id: str, ignore: Any = None) -> dict[str, Any]:  # noqa: A002
        if self.down and id == 'meta:config_revision':
            msg = 'configs unreachable'
            raise ConnectionError(msg)
        return await super().get(index, id, ignore)


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    fake = _ConfigsDown()
    registry = InMemoryRegistry()
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'env_model')
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'default'})
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.routers.curation.ingest_policy.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.app.resources.get_async_triton_pool', Pool)

    # The real dependency (it refreshes the project's config store) over a fake client.
    async def _make() -> Any:
        return fake

    monkeypatch.setattr('src.services.projects.guard.make_curation_opensearch', _make)
    from _curation_app import mount_curation_routers

    app = FastAPI()
    mount_curation_routers(app)
    with TestClient(app) as client:
        yield client, fake


def _activate_then_forget(client: TestClient, name: str, model: str) -> None:
    created = client.post(f'{BASE}/ingest_profiles', json={'name': name, 'body': body(model)})
    assert created.status_code == 201, created.text
    assert client.post(f'{BASE}/ingest_profiles/{name}/activate', json={}).status_code == 200
    reset_config_stores()  # a different worker process: nothing loaded yet


def test_class_sources_follow_an_activation_made_elsewhere(world: Any) -> None:
    client, _ = world
    _activate_then_forget(client, 'mine', 'birds')
    labels = {
        e['id']: e['label'] for e in client.get(f'{BASE}/class_sources').json()['class_sources']
    }
    assert labels['item_proposal'] == 'Proposed by birds'


def test_class_sources_refuse_when_the_store_is_unreadable(world: Any) -> None:
    client, fake = world
    fake.down = True
    resp = client.get(f'{BASE}/class_sources')
    assert resp.status_code == 503, resp.text
    assert resp.json()['detail']['error'] == 'config_store_unavailable'


def test_policy_put_refuses_when_the_store_is_unreadable(world: Any) -> None:
    client, fake = world
    fake.down = True
    resp = client.put(f'{BASE}/ingest/policy', json={'expected_revision': 0})
    assert resp.status_code == 503, resp.text
    assert resp.json()['detail']['error'] == 'config_store_unavailable'


@pytest.mark.parametrize('path', ['/models/status', '/regions/vocabulary', '/class_sources'])
def test_routes_that_read_the_ingest_profile_take_the_curation_client(path: str) -> None:
    from src.core.dependencies import get_curation_opensearch
    from src.routers.curation._mounting import curation_scoped_routers

    found = [
        route
        for router in curation_scoped_routers()
        for route in router.routes
        if getattr(route, 'path', '') == path and 'GET' in getattr(route, 'methods', ())
    ]
    assert found, path
    calls = {dep.call for dep in found[0].dependant.dependencies}  # type: ignore[attr-defined]
    assert get_curation_opensearch in calls

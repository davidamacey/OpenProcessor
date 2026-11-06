"""PUT /settings answers 409 for a stale revision: the defaults write used to be last-write-wins."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _config_fixtures import make_project
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine', 'api_async_engine')]

URL = '/curation/projects/default/settings'


@pytest.fixture(scope='module', autouse=True)
def _default_project(pg):
    return make_project(pg, 'default')


@pytest.fixture
def client(pg_env, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: object()  # noqa: PLW0108  # FastAPI reads a bare class as a dependency with parameters
    return TestClient(app)


def test_two_puts_with_the_same_revision_one_wins_and_one_is_409(client: TestClient) -> None:
    revision = client.get(URL).json()['revision']
    first = client.put(URL, json={'defaults': {'cluster': 'ahc'}, 'expected_revision': revision})
    second = client.put(URL, json={'defaults': {'cluster': 'ivf'}, 'expected_revision': revision})
    assert first.status_code == 200, first.text
    assert second.status_code == 409, second.text
    assert second.json()['detail']['error'] == 'revision_conflict'
    stored = client.get(URL).json()
    assert stored['defaults'] == {'cluster': 'ahc'}
    assert stored['revision'] == revision + 1


def test_a_put_without_expected_revision_is_refused(client: TestClient) -> None:
    assert client.put(URL, json={'defaults': {'cluster': 'ahc'}}).status_code == 422


def test_a_single_write_with_the_current_revision_succeeds(client: TestClient) -> None:
    revision = client.get(URL).json()['revision']
    done = client.put(URL, json={'defaults': {'cluster': 'ivf'}, 'expected_revision': revision})
    assert done.status_code == 200, done.text
    assert done.json()['revision'] == revision + 1

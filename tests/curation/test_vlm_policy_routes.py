"""`GET/PUT /vlm/policy` over an in-memory settings index."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import SettingsFakeOpenSearch
from src.config import IndexRole, get_curation_config, index_name


URL = '/curation/projects/default/vlm/policy'
INGEST_URL = '/curation/projects/default/ingest/policy'


@pytest.fixture
def fake() -> SettingsFakeOpenSearch:
    return SettingsFakeOpenSearch()


@pytest.fixture
def client(fake: SettingsFakeOpenSearch, monkeypatch: pytest.MonkeyPatch):
    async def _no_bootstrap(_: Any) -> None:
        return None

    monkeypatch.setattr('src.routers.curation.vlm_policy._ensure_indexes', _no_bootstrap)
    monkeypatch.setattr('src.routers.curation.ingest_policy._ensure_indexes', _no_bootstrap)
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def test_get_returns_the_defaults_when_never_written(client: TestClient) -> None:
    assert client.get(URL).json() == {
        'revision': 0,
        'scope': 'all',
        'conf_max': 0.8,
        'per_cluster': 5,
        'max_crops_per_day': 0,
        'sample_frac': 1.0,
    }


def test_put_round_trips_and_bumps_the_revision(
    client: TestClient, fake: SettingsFakeOpenSearch
) -> None:
    put = {'expected_revision': 0, 'scope': 'representatives', 'per_cluster': 3}
    first = client.put(URL, json=put)
    assert first.status_code == 200, first.text
    assert first.json()['revision'] == 1
    got = client.get(URL).json()
    assert (got['scope'], got['per_cluster'], got['conf_max']) == ('representatives', 3, 0.8)
    assert client.put(URL, json={'expected_revision': 1}).json()['scope'] == 'all'
    settings = fake.docs(index_name(get_curation_config(), IndexRole.SETTINGS))['default']
    assert settings['vlm_policy']['revision'] == 2


def test_stale_expected_revision_is_409_and_writes_nothing(
    client: TestClient, fake: SettingsFakeOpenSearch
) -> None:
    assert client.put(URL, json={'expected_revision': 0, 'scope': 'off'}).status_code == 200
    writes = fake.write_calls
    stale = client.put(URL, json={'expected_revision': 0, 'scope': 'all'})
    assert stale.status_code == 409
    assert stale.json()['detail']['error'] == 'revision_conflict'
    assert fake.write_calls == writes
    assert client.get(URL).json()['scope'] == 'off'


@pytest.mark.parametrize(
    'bad',
    [
        {'scope': 'sometimes'},
        {'sample_frac': 0.0},
        {'sample_frac': 1.5},
        {'conf_max': 1.2},
        {'per_cluster': 0},
        {'max_crops_per_day': -1},
        {'surprise': 1},
    ],
)
def test_invalid_bodies_are_422(client: TestClient, bad: dict[str, Any]) -> None:
    assert client.put(URL, json={'expected_revision': 0, **bad}).status_code == 422


def test_vlm_and_ingest_policies_do_not_clobber_each_other(client: TestClient) -> None:
    assert client.put(URL, json={'expected_revision': 0, 'scope': 'off'}).status_code == 200
    assert client.put(INGEST_URL, json={'expected_revision': 0}).status_code == 200
    assert client.get(URL).json()['scope'] == 'off'
    assert client.put(URL, json={'expected_revision': 1, 'scope': 'uncertain'}).status_code == 200
    assert client.get(INGEST_URL).json()['revision'] == 1

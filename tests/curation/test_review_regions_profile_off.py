"""``GET /review/regions`` with the region profile off serves an empty queue that
says why, not rows written under an earlier profile (GH #51)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import _common


P = f'{_common.config.api_prefix}/projects/default'


class _Recording:
    def __init__(self) -> None:
        self.searches = 0

    async def search(self, **_: Any) -> dict[str, Any]:
        self.searches += 1
        stale = {'_id': 'c1', '_source': {'crop_id': 'c1'}}
        return {'hits': {'total': {'value': 1}, 'hits': [stale]}}

    async def count(self, **_: Any) -> dict[str, Any]:
        return {'count': 1}

    async def get(self, **_: Any) -> dict[str, Any]:
        return {'found': True, '_source': {'crop_id': 'c1'}}


@pytest.fixture
def fake() -> _Recording:
    return _Recording()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake: _Recording) -> Any:
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', {'default'})
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


@pytest.fixture
def profile_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.detection.profile_registry.get_active_region_profile', lambda: None
    )


def test_regions_tab_is_an_explained_empty_queue_when_the_profile_is_off(
    client: TestClient, fake: _Recording, profile_off: None
) -> None:
    body = client.get(f'{P}/review/regions').json()
    assert (body['total'], body['items']) == (0, [])
    assert 'region profile is off' in body['empty_reason']
    assert fake.searches == 0


def test_locate_says_the_profile_is_off(client: TestClient, profile_off: None) -> None:
    body = client.get(f'{P}/review/regions/locate', params={'crop_id': 'c1'}).json()
    assert (body['in_queue'], body['reason']) == (False, 'region_profile_off')


def test_other_tabs_are_unaffected_by_the_profile_being_off(
    client: TestClient, fake: _Recording, profile_off: None
) -> None:
    body = client.get(f'{P}/review/all').json()
    assert body['total'] == 1
    assert fake.searches >= 1


@pytest.mark.usefixtures('reference_region_profile')
def test_regions_tab_serves_rows_when_a_profile_is_active(
    client: TestClient, fake: _Recording
) -> None:
    body = client.get(f'{P}/review/regions').json()
    assert body['total'] == 1
    assert fake.searches >= 1

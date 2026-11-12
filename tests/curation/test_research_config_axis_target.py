"""A cluster profile is activated only on the axis of its own target.

The route, the service and the axis check run for real; only the database is replaced, so a refused
activation is shown never to reach the repository's ``activate``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import uuid
from unittest.mock import AsyncMock

import pytest
from _curation_app import SCOPED, mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.db.repos import config as repo
from src.routers.curation import router as curation_router
from src.services.config_store import research_kinds


NOW = dt.datetime(2026, 11, 12, tzinfo=dt.UTC)


def _doc(kind: str, name: str, body: dict) -> repo.ConfigDoc:
    return repo.ConfigDoc(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        kind=kind,
        name=name,
        revision=3,
        body=body,
        description='',
        created_at=NOW,
        updated_at=NOW,
        updated_by='t',
        cloned_from_id=None,
        cloned_from_revision=None,
        current_revision=3,
        cloned_from=None,
    )


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict:
    docs: dict[str, repo.ConfigDoc] = {}
    activate = AsyncMock(
        side_effect=lambda _s, **kw: repo.ActiveRef(
            axis=kw['axis'], state='active', name='n', revision=kw['revision'], activated_at=NOW
        )
    )

    @contextlib.asynccontextmanager
    async def fake_tx(_kind: str):
        yield object(), uuid.uuid4()

    async def get_config(_s, *, name: str, **_kw):
        return docs.get(name)

    monkeypatch.setattr(research_kinds, '_tx', fake_tx)
    monkeypatch.setattr(repo, 'get_config', get_config)
    monkeypatch.setattr(repo, 'activate', activate)
    return {'docs': docs, 'activate': activate}


@pytest.fixture
def client(world: dict) -> TestClient:
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    return TestClient(app)


def _cluster(target: str) -> dict:
    return {'target': target}


def _activate(client: TestClient, name: str, axis: str):
    return client.post(
        f'{SCOPED}/cluster_profiles/{name}/activate',
        json={'axis': axis, 'revision': None, 'expected_active': None},
    )


def test_a_region_box_profile_on_the_item_axis_is_refused_with_a_served_code(
    client: TestClient, world: dict
) -> None:
    world['docs']['boxes'] = _doc('cluster_profile', 'boxes', _cluster('region_box'))
    r = _activate(client, 'boxes', 'clustering_item')
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    assert detail['error'] == 'axis_target_mismatch'
    assert detail['axis'] == 'clustering_item'
    assert "'boxes'" in detail['message']
    assert 'region_box' in detail['message']
    assert 'clustering_region_box' in detail['message']
    world['activate'].assert_not_called()


def test_an_item_profile_on_the_region_box_axis_is_refused(client: TestClient, world: dict) -> None:
    world['docs']['items'] = _doc('cluster_profile', 'items', _cluster('item'))
    r = _activate(client, 'items', 'clustering_region_box')
    assert r.status_code == 422
    assert r.json()['detail']['error'] == 'axis_target_mismatch'
    world['activate'].assert_not_called()


@pytest.mark.parametrize(
    ('target', 'axis'),
    [('item', 'clustering_item'), ('region_box', 'clustering_region_box')],
)
def test_a_profile_on_the_axis_of_its_target_is_activated(
    client: TestClient, world: dict, target: str, axis: str
) -> None:
    world['docs']['ok'] = _doc('cluster_profile', 'ok', _cluster(target))
    r = _activate(client, 'ok', axis)
    assert r.status_code == 200, r.text
    assert r.json()['axis'] == axis
    world['activate'].assert_called_once()


def test_kinds_whose_body_has_no_target_are_not_checked(client: TestClient, world: dict) -> None:
    world['docs']['strict'] = _doc('near_dup_profile', 'strict', {'version': 1})
    r = client.post(
        f'{SCOPED}/near_dup_profiles/strict/activate',
        json={'axis': 'near_dup', 'revision': None, 'expected_active': None},
    )
    assert r.status_code == 200, r.text

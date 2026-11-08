"""PUT /curation/projects/{project}/models/{name}/sharing (projects_plan.md
sec5.5, owner D1): opt-in cross-project model sharing, stored as ``model_shares`` rows.

The route wiring runs against the in-memory registry; the constraint, the lock and the
concurrency behaviour run against PostgreSQL in ``tests/promote/test_model_share_by_name.py``.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from _fake_model_registry import FakeModelRegistry

OWNER_ID = uuid.uuid4()
OTHERS = [
    SimpleNamespace(slug='wheels', id=uuid.uuid4()),
    SimpleNamespace(slug='boats', id=uuid.uuid4()),
]


@pytest.fixture(autouse=True)
def _writable(not_stale_registry: object) -> None:
    """PUT is a write; M2 refuses it on the process-wide default test
    registry, which is permanently stale by design (see the fixture)."""


class _Registry:
    async def ensure_fresh(self) -> None:
        return None

    def active_projects(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(slug='default', id=OWNER_ID), *OTHERS]


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch):
    import src.routers.curation._models_sharing as sharing_mod
    import src.routers.curation.models as models_mod
    from src.routers.curation import _raw_opensearch_dep

    monkeypatch.setattr(sharing_mod, 'current_project_id', lambda: OWNER_ID)
    monkeypatch.setattr(sharing_mod, 'get_project_registry', _Registry)
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, models_mod.router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: None
    with TestClient(app) as client:
        yield client


def test_sharing_opt_in_shares_with_every_other_project_and_bumps_the_revision(
    app_client: TestClient, model_registry_fake: FakeModelRegistry
) -> None:
    model_registry_fake.promote('cars_det_v3', owner='default', labels=['car'])

    r = app_client.put(
        '/curation/projects/default/models/cars_det_v3/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['shared'] is True
    assert body['revision'] == 2
    assert model_registry_fake.shares['cars_det_v3'] == {str(o.id) for o in OTHERS}


def test_sharing_stale_revision_409s(
    app_client: TestClient, model_registry_fake: FakeModelRegistry
) -> None:
    model_registry_fake.promote('cars_det_v3', owner='default', labels=['car'])

    r = app_client.put(
        '/curation/projects/default/models/cars_det_v3/sharing',
        json={'shared': True, 'expected_revision': 99},
    )
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'revision_conflict'
    assert r.json()['detail']['current_revision'] == 1


def test_sharing_on_a_model_this_project_does_not_own_404s(
    app_client: TestClient, model_registry_fake: FakeModelRegistry
) -> None:
    """Ownership itself is proven by test_promote_namespacing.py's project_owns_model unit
    tests; this only checks the route wires a 'not owned' result to 404."""
    model_registry_fake.promote('not_mine_det_v3', owner='cars', labels=['car'])

    r = app_client.put(
        '/curation/projects/default/models/not_mine_det_v3/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 404
    assert r.json()['detail']['error'] == 'model_not_found'
    assert r.json()['detail']['project'] == 'default'


def test_sharing_a_model_with_no_promotion_404s(app_client: TestClient) -> None:
    r = app_client.put(
        '/curation/projects/default/models/never_promoted/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 404
    assert r.json()['detail']['error'] == 'model_not_found'

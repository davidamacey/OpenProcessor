"""PUT /curation/projects/{project}/models/{name}/sharing (projects_plan.md
sec5.5, owner D1): opt-in cross-project model sharing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def app_client():
    import src.routers.curation.models as models_mod

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, models_mod.router)
    with TestClient(app) as client:
        yield client


def _promote(models_dir: Path, name: str, *, shared: bool = False) -> None:
    d = models_dir / name
    d.mkdir(parents=True)
    (d / 'promote.json').write_text(
        json.dumps(
            {
                'project': 'default',
                'shared': shared,
                'sharing_revision': 1,
                'classes': [{'model_id': 0, 'name': 'car'}],
            }
        )
    )


def test_sharing_opt_in_flips_shared_and_bumps_revision(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    _promote(tmp_path, 'cars_det_v3')

    r = app_client.put(
        '/curation/projects/default/models/cars_det_v3/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['shared'] is True
    assert body['revision'] == 2

    saved = json.loads((tmp_path / 'cars_det_v3' / 'promote.json').read_text())
    assert saved['shared'] is True
    assert saved['sharing_revision'] == 2


def test_sharing_stale_revision_409s(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    _promote(tmp_path, 'cars_det_v3')

    r = app_client.put(
        '/curation/projects/default/models/cars_det_v3/sharing',
        json={'shared': True, 'expected_revision': 99},
    )
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'revision_conflict'


def test_sharing_on_a_model_this_project_does_not_own_404s(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ownership itself is proven by test_promote_namespacing.py's
    _project_owns_model unit tests; this only checks the route wires a
    'not owned' result to 404."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    _promote(tmp_path, 'not_mine_det_v3')
    monkeypatch.setattr('src.routers.curation.models._project_owns_model', lambda _n: False)

    r = app_client.put(
        '/curation/projects/default/models/not_mine_det_v3/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 404


def test_sharing_a_model_with_no_promote_json_404s(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))

    r = app_client.put(
        '/curation/projects/default/models/never_promoted/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 404

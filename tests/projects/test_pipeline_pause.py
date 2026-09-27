"""POST /pause, POST /resume, GET /pause (projects_plan.md sec5.1/sec5.2):
the write side of the file-sentinel workers already read
(scripts/curation/_project_worker_utils.py's is_project_paused).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.curation._project_worker_utils import PIPELINE_PAUSED_FLAG_NAME, is_project_paused
from src.config.curation import base_curation_config
from src.config.projects import new_project_record


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def app_client():
    import src.routers.curation._project_pause as pause_mod

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, pause_mod.router)
    with TestClient(app) as client:
        yield client


def test_pause_then_resume_flips_the_flag_workers_read(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)

    r = app_client.get('/curation/projects/default/pause')
    assert r.status_code == 200
    assert r.json() == {'project': 'default', 'paused': False}

    r = app_client.post('/curation/projects/default/pause')
    assert r.status_code == 200
    assert r.json() == {'project': 'default', 'paused': True}

    # The exact flag the workers' is_project_paused() reads.
    record = new_project_record('default', base_curation_config())
    flag = record.resources.project_state_dir / PIPELINE_PAUSED_FLAG_NAME
    assert flag.exists()
    assert is_project_paused(record) is True

    r = app_client.get('/curation/projects/default/pause')
    assert r.json()['paused'] is True

    r = app_client.post('/curation/projects/default/resume')
    assert r.status_code == 200
    assert r.json() == {'project': 'default', 'paused': False}
    assert not flag.exists()
    assert is_project_paused(record) is False


def test_pause_and_resume_are_idempotent(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)

    assert app_client.post('/curation/projects/default/resume').status_code == 200
    assert app_client.post('/curation/projects/default/pause').status_code == 200
    assert app_client.post('/curation/projects/default/pause').status_code == 200

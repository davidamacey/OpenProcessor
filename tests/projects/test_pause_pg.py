"""POST /pause and /resume write ``projects.paused_at``; the workers read it from the registry
snapshot; no flag file is written anywhere."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from scripts.curation._project_worker_utils import is_project_paused, unpaused_projects
from src.services.projects.bootstrap import bootstrap_default_project
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import aseed_project


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]

PAUSE = '/curation/projects/default/pause'


@pytest.fixture
def registry() -> ProjectRegistry:
    async def _seed() -> ProjectRegistry:
        await bootstrap_default_project()
        await aseed_project('other')
        reg = ProjectRegistry()
        await reg.refresh_strict()
        return reg

    reg = asyncio.run(_seed())
    set_project_registry(reg)
    return reg


@pytest.fixture
def app_client(registry: ProjectRegistry):
    import src.routers.curation._project_pause as pause_mod

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, pause_mod.router)
    with TestClient(app) as client:
        yield client


def _worker_view(registry: ProjectRegistry) -> list[str]:
    async def _run() -> list[str]:
        return sorted(r.slug for r in await unpaused_projects(registry, None))

    return asyncio.run(_run())


def test_pause_makes_the_worker_skip_the_project_and_resume_clears_it(
    app_client: TestClient, registry: ProjectRegistry, tmp_path: Path
) -> None:
    assert _worker_view(registry) == ['default', 'other']
    assert app_client.get(PAUSE).json() == {
        'project': 'default',
        'paused': False,
        'paused_by': [],
        'reason': None,
    }

    resp = app_client.post(PAUSE)
    assert resp.status_code == 200, resp.text
    assert resp.json()['paused'] is True
    assert resp.json()['paused_by'] == ['project']
    assert _worker_view(registry) == ['other']  # the others keep running
    record = registry.get('default')
    assert record is not None
    assert record.paused_at is not None
    assert is_project_paused(record)
    assert app_client.get(PAUSE).json()['paused'] is True

    resp = app_client.post('/curation/projects/default/resume')
    assert resp.status_code == 200, resp.text
    assert resp.json()['paused'] is False
    assert _worker_view(registry) == ['default', 'other']
    assert not is_project_paused(registry.get('default'))  # type: ignore[arg-type]


def test_pause_writes_no_flag_file(app_client: TestClient, registry: ProjectRegistry) -> None:
    assert app_client.post(PAUSE).status_code == 200
    record = registry.get('default')
    assert record is not None
    state_dir = record.resources.project_state_dir
    assert not (state_dir / 'pipeline_paused.flag').exists()
    assert not list(state_dir.glob('*paused*')) if state_dir.exists() else True


def test_pause_and_resume_are_idempotent(app_client: TestClient) -> None:
    assert app_client.post('/curation/projects/default/resume').status_code == 200
    assert app_client.post(PAUSE).status_code == 200
    assert app_client.post(PAUSE).status_code == 200


def test_pausing_one_project_leaves_the_others_paused_state_alone(
    app_client: TestClient, registry: ProjectRegistry
) -> None:
    app_client.post(PAUSE)
    asyncio.run(registry.refresh_strict())
    other = registry.get('other')
    assert other is not None
    assert other.paused_at is None


def test_pause_state_reports_global_gpu_training_claim(app_client: TestClient, pg) -> None:
    """A running job that holds cards (the arbiter tick's record) idles every project's workers."""
    from fixtures.job_rows import sql

    from src.domain import gpu_arbiter_state as state

    assert app_client.get('/curation/projects/default/pause').json()['paused'] is False
    sql(
        pg,
        'INSERT INTO deployment_settings (key, value, updated_by) VALUES (%s, %s, %s) '
        'ON CONFLICT (key) DO UPDATE SET value = excluded.value',
        state.KEY,
        Jsonb({state.ARBITER: state.ArbiterRecord(claims=('job-1',)).to_json()}),
        'test',
    )
    try:
        r = app_client.get('/curation/projects/default/pause')
        assert r.status_code == 200
        body = r.json()
        assert body['paused'] is True
        assert body['paused_by'] == ['gpu_training']
        assert body['reason'] == 'GPU claim active (1 running job)'
    finally:
        sql(pg, 'DELETE FROM deployment_settings WHERE key = %s', state.KEY)


def test_pause_publishes_a_global_event(
    app_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_events: list[dict],
) -> None:
    import src.config.curation_accessors as curation_config_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)

    assert app_client.post('/curation/projects/default/pause').status_code == 200
    assert app_client.post('/curation/projects/default/resume').status_code == 200
    assert [(e['type'], e['target'], e['project']) for e in captured_events] == [
        ('project.paused', 'default', None),
        ('project.resumed', 'default', None),
    ]

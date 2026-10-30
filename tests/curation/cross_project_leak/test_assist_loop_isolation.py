"""Auto-label loops belong to one project: its job directory, its state, its hold on retraining."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import SCOPED
from src.config.project_context import ProjectDirMismatchError, bind_project
from src.schemas.curation.assist_loop import AssistLoopRequest
from src.services.curation.assist_loop import job


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


def _state() -> dict[str, Any]:
    return {
        'status': 'queued',
        'request': AssistLoopRequest().model_dump(mode='json'),
        'pinned': {'vlm': {'name': 'v'}, 'prompt_pack': {'name': 'p'}},
        'plan': {
            'roots': 0,
            'eligible_items': 0,
            'required_sample_table': [],
            'estimated_vlm_crops_root_level': 0,
            'estimated_vlm_requests_root_level': 0,
        },
    }


def _url(slug: str, path: str) -> str:
    return f'{SCOPED.format(project=slug)}/assist/loops{path}'


@pytest.fixture
def alpha_loop(leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv('OP_ASSIST_LOOP_JOBS_DIR', str(leak_env.root / 'loops'))
    with bind_project(leak_env.records['alpha']):
        return job.create_loop(state=_state(), request={}).directory.name


def test_a_loop_is_served_to_its_own_project_only(leak_env: LeakEnv, alpha_loop: str) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)

    assert client.get(_url('alpha', f'/{alpha_loop}')).status_code == 200
    assert [x['loop_id'] for x in client.get(_url('alpha', '')).json()['loops']] == [alpha_loop]
    assert client.get(_url('beta', '')).json() == {'loops': []}
    for method, path in (
        ('get', f'/{alpha_loop}'),
        ('get', f'/{alpha_loop}/nodes'),
        ('get', f'/{alpha_loop}/nodes/r10000/members'),
        ('post', f'/{alpha_loop}/cancel'),
        ('post', f'/{alpha_loop}/resume'),
        ('post', f'/{alpha_loop}/rollback'),
    ):
        assert getattr(client, method)(_url('beta', path)).status_code == 404, (method, path)
    with bind_project(leak_env.records['alpha']):
        assert job.open_job(alpha_loop).read()['status'] == 'queued', 'beta cancelled nothing'


def test_a_live_loop_holds_only_its_own_projects_retraining(
    leak_env: LeakEnv, alpha_loop: str
) -> None:
    with bind_project(leak_env.records['alpha']), pytest.raises(Exception, match='running'):
        job.require_no_live_loop()
    with bind_project(leak_env.records['beta']):
        job.require_no_live_loop()
        assert job.loop_live() is False


def test_a_loops_directory_is_marked_with_its_project(leak_env: LeakEnv, alpha_loop: str) -> None:
    with bind_project(leak_env.records['alpha']):
        alpha_root = job.jobs_root()
        assert (alpha_root / '.project').read_text() == 'alpha'
    with bind_project(leak_env.records['beta']):
        beta_root = job.jobs_root()
    assert beta_root != alpha_root
    shutil.copytree(alpha_root, beta_root, dirs_exist_ok=True)
    with bind_project(leak_env.records['beta']), pytest.raises(ProjectDirMismatchError):
        job.live_loop_ids()

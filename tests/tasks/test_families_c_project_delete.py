"""``project_delete`` through the whole job path: queue, claim, handler, rows, event, retired slug.

A fake producer stands in for the broker; Celery's ``Task.apply`` runs the registered task with
the real wrapper (https://docs.celeryq.dev/en/stable/userguide/testing.html).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fixtures.job_rows import sql
from projects.conftest import FakeLifecycleOpenSearch, fake_ensure_indexes

from src.db.session import async_session_factory
from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation_accessors as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(ProjectRegistry())
    with patch(
        'src.services.curation.index_bootstrap.ensure_indexes',
        new=AsyncMock(side_effect=fake_ensure_indexes),
    ):
        yield
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


async def _two_projects(client):
    await lifecycle.create_project(client, slug='alpha', display_name='Alpha')
    await lifecycle.create_project(client, slug='beta', display_name='Beta')
    deleting = await lifecycle.delete_project(slug='alpha', confirm='alpha')
    async with async_session_factory()() as session:
        started = await lifecycle.start_project_delete(session, 'alpha')
    return deleting, started


def test_the_registered_task_runs_the_finish_once_and_the_slug_is_retired(
    pg, job_env, producer, wakes, kinds, worker_app, queue_env, monkeypatch
) -> None:
    client = FakeLifecycleOpenSearch()
    monkeypatch.setattr(
        'src.services.projects.guard.make_script_opensearch', lambda *_a, **_k: client
    )
    deleting, started = asyncio.run(_two_projects(client))
    job_id = started.job.id
    assert deleting.status == 'deleting'
    assert producer.sent() == [str(job_id)]
    [(project_id, scope, status)] = sql(
        pg, 'SELECT project_id, scope, status FROM jobs WHERE id = %s', job_id
    )
    assert (project_id, scope, status) == (None, 'alpha', 'queued')

    async def run() -> None:
        outcome = await asyncio.to_thread(
            worker_app.tasks['op.families_c.project_delete'].apply,
            args=[str(job_id)],
            task_id=str(job_id),
        )
        assert outcome.successful(), outcome.traceback

    asyncio.run(run())

    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', job_id)
    assert status == 'completed'
    assert result['project'] == 'alpha'
    assert sql(pg, "SELECT 1 FROM projects WHERE slug = 'alpha'") == []
    assert sql(pg, "SELECT retired_at IS NOT NULL FROM project_slugs WHERE slug = 'alpha'") == [
        (True,)
    ]
    assert sql(pg, "SELECT 1 FROM projects WHERE slug = 'beta'") == [(1,)]
    assert sorted(client.deleted_indexes), 'the finish removed the project indexes once'
    deleted = sql(pg, "SELECT payload->>'target' FROM events WHERE type = 'project.deleted'")
    assert deleted == [('alpha',)]


def test_the_finish_is_idempotent_a_rerun_after_completion_changes_nothing(
    pg, job_env, producer, wakes, kinds, monkeypatch
) -> None:
    """A reaper resume (or a second job) finds the slug retired and answers without work."""
    from src.tasks import jobs_c

    client = FakeLifecycleOpenSearch()
    monkeypatch.setattr(
        'src.services.projects.guard.make_script_opensearch', lambda *_a, **_k: client
    )
    asyncio.run(_two_projects(client))

    class Ctx:
        params = {'slug': 'alpha'}

        async def client(self, _name):
            return client

        async def checkpoint(self, _cursor):
            return None

        async def progress(self, *_a, **_k):
            return None

    first = asyncio.run(jobs_c.project_delete(Ctx()))
    removed = list(client.deleted_indexes)
    second = asyncio.run(jobs_c.project_delete(Ctx()))
    assert first['project'] == second['project'] == 'alpha'
    assert client.deleted_indexes == removed
    assert sql(pg, "SELECT 1 FROM projects WHERE slug = 'alpha'") == []

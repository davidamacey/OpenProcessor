"""#439: ``POST /curation/projects`` queues a ``project_create`` job instead of building inline.

The route claims the slug (a ``building`` row) and creates no index; the registered task builds the
project and ends it ``active`` (or ``failed``). A fake producer stands in for the broker and
Celery's ``Task.apply`` runs the task with the real wrapper
(https://docs.celeryq.dev/en/stable/userguide/testing.html).
"""

# ruff: noqa: F811 - fixtures are imported for registration and requested by name

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Response
from fixtures.job_fixtures import job_env, kinds, producer, wakes  # noqa: F401  # fixture registration
from fixtures.job_rows import sql
from projects.conftest import FakeLifecycleOpenSearch, fake_ensure_indexes

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.routers.curation import projects as projects_router
from src.schemas.curation.projects import CreateProjectRequest
from src.services.projects.registry import ProjectRegistry, set_project_registry


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


@pytest.fixture(autouse=True)
def fake_os(tmp_path, monkeypatch):
    import src.config.curation_accessors as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(ProjectRegistry())
    client = FakeLifecycleOpenSearch()

    async def _make():
        return client

    monkeypatch.setattr(projects_router, 'make_curation_opensearch', _make)
    monkeypatch.setattr(
        'src.services.projects.guard.make_script_opensearch', lambda *_a, **_k: client
    )
    with patch(
        'src.services.curation.index_bootstrap.ensure_indexes',
        new=AsyncMock(side_effect=fake_ensure_indexes),
    ):
        yield client
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


async def _post(slug: str, key: str | None = None):
    response = Response()
    async with async_session_factory()() as session:
        body = await projects_router.create_project(
            CreateProjectRequest(slug=slug, display_name=slug.title()), response, session, key
        )
    return body, response


def _run(worker_app, job_id: str):
    async def go():
        return await asyncio.to_thread(
            worker_app.tasks['op.b4_project.project_create'].apply,
            args=[job_id],
            task_id=job_id,
        )

    return asyncio.run(go())


def test_the_route_queues_a_job_and_creates_no_index(
    pg, job_env, producer, wakes, kinds, fake_os
) -> None:
    body, response = asyncio.run(_post('cars'))
    assert response.status_code == 202
    assert body.project.status == 'building'
    assert body.job_id == producer.sent()[0]
    assert fake_os.indexes == {}, 'the handler created an index'
    assert sql(pg, "SELECT status FROM projects WHERE slug = 'cars'") == [('building',)]
    [(kind, scope, status)] = sql(
        pg, 'SELECT kind, scope, status FROM jobs WHERE id = %s', body.job_id
    )
    assert (kind, scope, status) == ('project_create', 'cars', 'queued')


def test_a_taken_slug_is_409_before_any_enqueue(pg, job_env, producer, wakes, kinds) -> None:
    asyncio.run(_post('cars'))
    with pytest.raises(RequestRejectedError) as taken:
        asyncio.run(_post('cars'))
    assert taken.value.detail['error'] == 'slug_taken'
    assert len(producer.sent()) == 1
    assert sql(pg, "SELECT count(*) FROM jobs WHERE kind = 'project_create'") == [(1,)]


def test_a_repeat_with_the_same_key_answers_the_first_job(
    pg, job_env, producer, wakes, kinds
) -> None:
    first, _ = asyncio.run(_post('cars', 'create-cars-key-1'))
    again, response = asyncio.run(_post('cars', 'create-cars-key-1'))
    assert response.status_code == 200
    assert again.job_id == first.job_id
    assert again.project.status == 'building'
    assert len(producer.sent()) == 1


def test_the_task_builds_the_project_and_ends_it_active(
    pg, job_env, producer, wakes, kinds, worker_app, fake_os
) -> None:
    body, _ = asyncio.run(_post('cars'))
    outcome = _run(worker_app, body.job_id)
    assert outcome.successful(), outcome.traceback
    assert sql(pg, "SELECT status FROM projects WHERE slug = 'cars'") == [('active',)]
    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', body.job_id)
    assert status == 'completed'
    assert result['project'] == 'cars'
    assert {'project_create_indexes', 'project_create_finish'} <= set(result['stages'])
    assert fake_os.indexes, 'the task created the indexes'
    assert sql(pg, "SELECT payload->>'target' FROM events WHERE type = 'project.created'") == [
        ('cars',)
    ]


def test_a_failing_build_leaves_the_project_failed_and_the_job_failed(
    pg, job_env, producer, wakes, kinds, worker_app
) -> None:
    body, _ = asyncio.run(_post('cars'))
    with patch(
        'src.services.curation.index_bootstrap.ensure_indexes',
        new=AsyncMock(side_effect=RuntimeError('opensearch is down')),
    ):
        _run(worker_app, body.job_id)
    assert sql(pg, "SELECT status FROM projects WHERE slug = 'cars'") == [('failed',)]
    [(status,)] = sql(pg, 'SELECT status FROM jobs WHERE id = %s', body.job_id)
    assert status in ('failed', 'queued')  # a failed attempt may be queued for the next one
    assert sql(pg, "SELECT 1 FROM events WHERE type = 'project.created'") == []

"""``/projects/combine*`` over HTTP and the queue: preview, start, the task, poll, cancel and
resume, and the target's status transitions (4B-4)."""

from __future__ import annotations

import asyncio
import sys
import types
import uuid
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fixtures.job_rows import sql

from src.config.project_context import current_project
from src.routers.curation import projects_combine
from src.routers.curation._project_deps import install_project_exception_handlers
from src.routers.curation.projects import global_router
from src.services.projects import lifecycle
from src.services.projects.combine import execute

from .test_combine_execute import MAPPING, build


if TYPE_CHECKING:
    from .world import World

pytestmark = pytest.mark.db


def _body(**extra: Any) -> dict[str, Any]:
    return {
        'target': {'slug': 'combined', 'display_name': 'Combined'},
        'sources': [{'project': 'cars-a'}, {'project': 'cars-b'}],
        'class_mapping': MAPPING,
        **extra,
    }


@pytest_asyncio.fixture
async def api(world: World, pg_registry, job_env, producer, wakes, monkeypatch):
    world.pg = pg_registry
    build(world)
    calls: dict[str, list[Any]] = {'create': [], 'finish': [], 'pending': [], 'bound': []}

    async def make_client() -> Any:
        return world.fake

    async def create_project(client: Any, **kwargs: Any):
        calls['create'].append(kwargs)
        return world.project(kwargs['slug'], [], status='building'), []

    async def finish_building(client: Any, record: Any, *, ok: bool = True):
        calls['finish'].append((record.slug, ok))
        return record

    async def enqueue_pending(session: Any, project: Any, level: str, ids: list[str], reason: str):
        calls['pending'].append((project, level, len(ids), reason))

    real_copy = execute.copy_page

    async def spying_copy(ctx: Any, source_index: int, page: Any) -> Any:
        calls['bound'].append(current_project().record.slug)
        return await real_copy(ctx, source_index, page)

    package = types.ModuleType('src.services.near_duplicates')
    pending = types.ModuleType('src.services.near_duplicates.pending')
    pending.enqueue_pending = enqueue_pending  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'src.services.near_duplicates', package)
    monkeypatch.setitem(sys.modules, 'src.services.near_duplicates.pending', pending)
    monkeypatch.setattr(execute, 'copy_page', spying_copy)
    monkeypatch.setattr(projects_combine, 'make_curation_opensearch', make_client)
    monkeypatch.setattr(lifecycle, 'create_project', create_project)
    monkeypatch.setattr(lifecycle, 'finish_building', finish_building)
    monkeypatch.setattr(
        'src.services.projects.guard.make_script_opensearch', lambda *_a, **_k: world.fake
    )
    app = FastAPI()
    app.include_router(global_router, prefix='/curation')
    install_project_exception_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        client.calls = calls  # type: ignore[attr-defined]
        yield client


async def _start(api: httpx.AsyncClient) -> str:
    preview = await api.post('/curation/projects/combine/preview', json=_body())
    assert preview.status_code == 200
    body = preview.json()
    assert body['ok'] is True
    assert body['dedup']['identical_images'] == 1
    started = await api.post(
        '/curation/projects/combine', json=_body(expected_preview_sha=body['preview_sha'])
    )
    assert started.status_code == 202, started.text
    assert started.json()['target'] == 'combined'
    return started.json()['job_id']


async def _run_task(worker_app: Any, job_id: str) -> None:
    outcome = await asyncio.to_thread(
        worker_app.tasks['op.families_b.combine'].apply, args=[job_id], task_id=job_id
    )
    assert outcome.successful(), outcome.traceback


@pytest.mark.asyncio
async def test_start_inserts_one_queued_job_with_its_rows_and_sends_one_message(
    api: httpx.AsyncClient, world: World, pg, producer
) -> None:
    job_id = await _start(api)
    assert producer.sent() == [job_id]
    target = world.records['combined']
    rows = sql(pg, 'SELECT kind, status, project_id FROM jobs WHERE id = %s', uuid.UUID(job_id))
    assert rows == [('combine', 'queued', target.id)]
    assert sql(pg, 'SELECT target_project_id, holdout_mode FROM combine_runs') == [
        (target.id, 'union')
    ]
    assert sql(pg, 'SELECT position, source_slug FROM combine_sources ORDER BY position') == [
        (0, 'cars-a'),
        (1, 'cars-b'),
    ]
    assert api.calls['create'][0]['origin']['kind'] == 'combine'  # type: ignore[attr-defined]
    assert api.calls['create'][0]['activate'] is False  # type: ignore[attr-defined]
    status = (await api.get(f'/curation/projects/combine/{job_id}')).json()
    assert status['status'] == 'queued'
    assert status['sources'] == ['cars-a', 'cars-b']
    assert status['target'] == 'combined'


@pytest.mark.asyncio
async def test_a_broker_failure_fails_the_row_and_the_target(
    api: httpx.AsyncClient, pg, producer
) -> None:
    producer.fail_send = True
    preview = (await api.post('/curation/projects/combine/preview', json=_body())).json()
    reply = await api.post(
        '/curation/projects/combine', json=_body(expected_preview_sha=preview['preview_sha'])
    )
    assert reply.status_code == 503
    assert reply.json()['detail']['error'] == 'queue_unavailable'
    assert sql(pg, 'SELECT status FROM jobs') == [('failed',)]
    assert api.calls['finish'] == [('combined', False)]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_the_task_runs_in_the_target_scope_and_activates_it(
    api: httpx.AsyncClient, world: World, producer, worker_app
) -> None:
    job_id = await _start(api)
    await _run_task(worker_app, job_id)

    job = (await api.get(f'/curation/projects/combine/{job_id}')).json()
    assert job['status'] == 'completed', job
    assert job['done'] == job['total'] == 6  # every source image, the duplicate included
    assert job['report']['images_copied'] == 5
    assert job['next_steps'][0]['action'] == 'recluster'
    assert set(api.calls['bound']) == {'combined'}  # type: ignore[attr-defined]
    assert api.calls['finish'] == [('combined', True)]  # type: ignore[attr-defined]
    assert len(world.images('combined')) == 5
    levels = {level for _p, level, _n, reason in api.calls['pending'] if reason == 'added'}  # type: ignore[attr-defined]
    assert levels == {'image', 'item'}
    assert sum(n for _p, level, n, _r in api.calls['pending'] if level == 'image') == 5  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_stale_preview_sha_is_409(api: httpx.AsyncClient) -> None:
    r = await api.post('/curation/projects/combine', json=_body(expected_preview_sha='deadbeef'))
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'preview_stale'
    assert api.calls['create'] == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_an_invalid_request_is_422_with_a_report(api: httpx.AsyncClient) -> None:
    body = _body(expected_preview_sha='x')
    body['class_mapping'] = {}
    r = await api.post('/curation/projects/combine', json=body)
    assert r.status_code == 422
    detail = r.json()['detail']
    assert detail['error'] == 'combine_invalid'
    assert {e['code'] for e in detail['report']['errors']} == {'unmapped_class'}


@pytest.mark.asyncio
async def test_a_to_or_drop_mapping_body_is_422(api: httpx.AsyncClient) -> None:
    body = _body()
    body['class_mapping'] = {'cars-a': [{'dataset_class': 'car', 'to': 'vehicle'}]}
    assert (await api.post('/curation/projects/combine/preview', json=body)).status_code == 422
    body['class_mapping'] = {'cars-a': [{'dataset_class': 'car', 'drop': True}]}
    assert (await api.post('/curation/projects/combine/preview', json=body)).status_code == 422


@pytest.mark.asyncio
async def test_unknown_jobs_and_wrong_state_actions(
    api: httpx.AsyncClient, worker_app, producer
) -> None:
    unknown = str(uuid.uuid4())
    assert (await api.get(f'/curation/projects/combine/{unknown}')).status_code == 404
    assert (await api.get('/curation/projects/combine/not-a-job-id')).status_code == 404
    assert (await api.post('/curation/projects/combine/not-a-job-id/cancel')).status_code == 404
    job_id = await _start(api)
    await _run_task(worker_app, job_id)
    cancel = await api.post(f'/curation/projects/combine/{job_id}/cancel')
    assert cancel.status_code == 409
    assert cancel.json()['detail']['error'] == 'combine_not_resumable'
    resume = await api.post(f'/curation/projects/combine/{job_id}/resume')
    assert resume.status_code == 409


@pytest.mark.asyncio
async def test_cancel_while_queued_cancels_the_job(api: httpx.AsyncClient, producer) -> None:
    job_id = await _start(api)
    cancelled = await api.post(f'/curation/projects/combine/{job_id}/cancel')
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()['status'] == 'cancelled'


async def _interrupt(pg, job_id: str) -> None:
    sql(
        pg,
        "UPDATE jobs SET status = 'running', owner = 'test', "
        "lease_expires_at = now() + interval '1 hour', started_at = now() WHERE id = %s",
        uuid.UUID(job_id),
    )
    sql(pg, "UPDATE jobs SET status = 'interrupted' WHERE id = %s", uuid.UUID(job_id))


@pytest.mark.asyncio
async def test_an_interrupted_combine_resumes_under_the_same_id(
    api: httpx.AsyncClient, world: World, pg, producer
) -> None:
    job_id = await _start(api)
    await _interrupt(pg, job_id)
    world.records['combined'] = world.records['combined']  # still building
    resumed = await api.post(f'/curation/projects/combine/{job_id}/resume')
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()['status'] == 'queued'
    assert producer.sent() == [job_id, job_id]


@pytest.mark.asyncio
async def test_resume_refuses_when_a_source_changed_since_the_preview(
    api: httpx.AsyncClient, world: World, pg, producer
) -> None:
    job_id = await _start(api)
    await _interrupt(pg, job_id)
    world.add_image('cars-a', items=[{'cls': 'car', 'bbox': [0.1, 0.1, 0.5, 0.5]}])
    resumed = await api.post(f'/curation/projects/combine/{job_id}/resume')
    assert resumed.status_code == 409
    assert resumed.json()['detail']['error'] == 'preview_stale'
    assert producer.sent() == [job_id]


@pytest.mark.asyncio
async def test_resume_refuses_when_the_target_is_no_longer_building(
    api: httpx.AsyncClient, world: World, pg, producer
) -> None:
    import dataclasses

    job_id = await _start(api)
    await _interrupt(pg, job_id)
    world.records['combined'] = dataclasses.replace(world.records['combined'], status='active')
    resumed = await api.post(f'/curation/projects/combine/{job_id}/resume')
    assert resumed.status_code == 409
    assert resumed.json()['detail']['error'] == 'combine_not_resumable'


def test_combine_is_not_a_project_slug_route() -> None:
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(global_router, prefix='/curation')
    install_project_exception_handlers(app)
    # `combine` is reserved, so these never fall through to /projects/{project}.
    assert TestClient(app).get('/curation/projects/combine/preview').status_code in (404, 405)

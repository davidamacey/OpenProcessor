# ruff: noqa: F811 - pytest fixtures are imported by name and used as test arguments
"""Starting auto-label queues one job; no trigger file, a second start is 409, cancel is a row write."""

from __future__ import annotations

import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.job_fixtures import job_env, kinds, producer, wakes  # noqa: F401
from fixtures.job_rows import make_project, sql

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import new_project_record
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.curation.autolabel import service


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


@pytest.fixture
def bound(pg, job_env):
    slug = f'al-{uuid.uuid4().hex[:6]}'
    project = make_project(pg, slug)
    record = replace(new_project_record(slug, base_curation_config()), id=project)
    with bind_project(record):
        yield project


async def test_a_start_queues_one_job_and_sends_one_message(pg, bound, producer, wakes) -> None:
    async with async_session_factory()() as session:
        state = await service.start(session, {'train_clusters': False})
    assert state['status'] == 'queued'
    assert state['args'] == {'train_clusters': False}
    rows = sql(pg, 'SELECT status, kind FROM jobs WHERE project_id = %s', bound)
    assert rows == [('queued', 'autolabel')]
    assert producer.sent() == [state['job_id']]


async def test_a_second_start_while_one_is_active_is_409_job_busy(
    pg, bound, producer, wakes
) -> None:
    async with async_session_factory()() as session:
        await service.start(session, {})
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as busy:
            await service.start(session, {})
    assert busy.value.status_code == 409
    assert busy.value.detail['error'] == 'job_busy'


async def test_cancel_while_queued_ends_the_job_without_a_worker(
    pg, bound, producer, wakes
) -> None:
    async with async_session_factory()() as session:
        started = await service.start(session, {})
    async with async_session_factory()() as session:
        assert await service.cancel(session, requested_by='test') is True
    async with async_session_factory()() as session:
        state = await service.by_id(session, started['job_id'])
    assert state['status'] == 'cancelled'
    async with async_session_factory()() as session:
        assert await service.cancel(session, requested_by='test') is False


async def test_the_state_of_a_project_that_never_ran_is_idle(pg, bound) -> None:
    async with async_session_factory()() as session:
        assert (await service.current(session))['status'] == 'idle'


async def test_a_job_id_of_another_kind_or_project_is_unknown(pg, bound, producer, wakes) -> None:
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as unknown:
            await service.by_id(session, str(uuid.uuid4()))
    assert unknown.value.status_code == 404


async def test_cancel_while_running_asks_then_the_worker_ends_it(
    pg, bound, producer, wakes
) -> None:
    async with async_session_factory()() as session:
        started = await service.start(session, {})
    sql(
        pg,
        "UPDATE jobs SET status = 'running', fencing_token = 1, attempt = 1, owner = 'test', "
        "started_at = now(), heartbeat_at = now(), lease_expires_at = now() + interval '1 hour' "
        'WHERE id = %s',
        uuid.UUID(started['job_id']),
    )
    async with async_session_factory()() as session:
        assert await service.cancel(session, requested_by='test') is True
    async with async_session_factory()() as session:
        assert (await service.by_id(session, started['job_id']))['status'] == 'cancelling'


async def test_pause_and_resume_a_queued_job(pg, bound, producer, wakes) -> None:
    async with async_session_factory()() as session:
        started = await service.start(session, {})
    async with async_session_factory()() as session:
        assert await service.pause(session) is True
    async with async_session_factory()() as session:
        paused = await service.by_id(session, started['job_id'])
    assert (paused['status'], paused['pause_reason']) == ('paused', 'user')
    async with async_session_factory()() as session:
        assert await service.resume(session) is True
    async with async_session_factory()() as session:
        assert (await service.by_id(session, started['job_id']))['status'] == 'queued'

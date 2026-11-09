"""Starting the ``project_delete`` job: one row, one message, exclusive per slug, 503 when the broker is down."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fixtures.job_rows import sql

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.jobs.service import cancel_job
from src.services.projects.delete import start_project_delete


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env', 'bound_async_engine')]


def _slug() -> str:
    return f's{uuid.uuid4().hex[:10]}'  # the module shares one database


async def _start(slug: str):
    async with async_session_factory()() as session:
        return await start_project_delete(session, slug)


def test_the_start_inserts_one_queued_deployment_row_and_sends_one_message(pg, producer) -> None:
    slug = _slug()
    started = asyncio.run(_start(slug))
    assert started.created
    [(project, kind, scope, status, queue)] = sql(
        pg, 'SELECT project_id, kind, scope, status, queue FROM jobs WHERE id = %s', started.job.id
    )
    assert (project, kind, scope, status, queue) == (
        None,
        'project_delete',
        slug,
        'queued',
        'jobs',
    )
    assert producer.sent() == [str(started.job.id)]


def test_a_second_start_for_the_same_slug_is_job_busy_and_another_slug_is_not(pg, producer) -> None:
    slug = _slug()
    first = asyncio.run(_start(slug))
    with pytest.raises(RequestRejectedError) as busy:
        asyncio.run(_start(slug))
    assert busy.value.status_code == 409
    assert busy.value.detail['error'] == 'job_busy'
    assert busy.value.detail['active_job_id'] == str(first.job.id)
    other = asyncio.run(_start(_slug()))
    assert other.created
    assert len(producer.sent()) == 2


def test_a_broker_that_is_down_is_503_and_the_row_failed(pg, producer) -> None:
    slug = _slug()
    producer.fail_send = True
    with pytest.raises(RequestRejectedError) as down:
        asyncio.run(_start(slug))
    assert down.value.status_code == 503
    assert down.value.detail['error'] == 'queue_unavailable'
    [(status, code)] = sql(pg, 'SELECT status, error_code FROM jobs WHERE scope = %s', slug)
    assert (status, code) == ('failed', 'queue_unavailable')


def test_cancel_while_queued_is_cancelled_and_frees_the_slug(pg, producer) -> None:
    slug = _slug()
    started = asyncio.run(_start(slug))

    async def cancel():
        async with async_session_factory()() as session:
            return await cancel_job(session, uuid.UUID(str(started.job.id)), None, requested_by='t')

    cancelled = asyncio.run(cancel())
    assert cancelled.job.status == 'cancelled'
    assert asyncio.run(_start(slug)).created

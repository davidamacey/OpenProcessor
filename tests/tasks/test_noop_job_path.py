"""The whole job path with the real ``op.infra.noop`` task: start, dispatch, claim, finish.

A fake producer stands in for the broker; Celery's ``Task.apply`` runs the registered task with
the real signals and the real wrapper (https://docs.celeryq.dev/en/stable/userguide/testing.html).
"""

from __future__ import annotations

import pytest
from fixtures.job_rows import sql

from src.db.session import async_session_factory
from src.services.jobs.service import start_job


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


@pytest.mark.asyncio
async def test_a_started_noop_is_dispatched_and_completes_with_its_state_events(
    pg, job_env, producer, wakes, kinds, worker_app, queue_env
) -> None:
    async with async_session_factory()() as session:
        started = await start_job(session, 'noop', None)
    job_id = started.job.id
    assert producer.sent() == [str(job_id)]
    assert producer.calls[0] == ('send', str(job_id))

    import asyncio

    outcome = await asyncio.to_thread(
        worker_app.tasks['op.infra.noop'].apply, args=[str(job_id)], task_id=str(job_id)
    )
    assert outcome.successful(), outcome.traceback

    [(status, result, attempt)] = sql(
        pg, 'SELECT status, result, attempt FROM jobs WHERE id = %s', job_id
    )
    assert (status, result, attempt) == ('completed', {'noop': True}, 1)
    states = sql(
        pg,
        "SELECT payload->>'status', project_id FROM events WHERE job_id = %s AND type = 'job.state' "
        'ORDER BY id',
        job_id,
    )
    assert states == [('queued', None), ('running', None), ('completed', None)]  # the global stream

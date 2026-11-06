"""A queued job pauses at once; its stale message is acked; resuming dispatches the same id."""

from __future__ import annotations

import asyncio

import pytest
from fixtures.job_rows import make_project, sql

from src.db.session import async_session_factory
from src.services.jobs.service import pause_job, resume_job, start_job
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db


async def run(fn, *args, **kwargs):
    async with async_session_factory()() as session:
        return await fn(session, *args, **kwargs)


@pytest.mark.asyncio
async def test_pause_the_stale_message_and_resume(pg, job_env, producer, wakes, kinds) -> None:
    project = make_project(pg)
    started = await run(start_job, 't_pause', project)
    job = started.job
    paused = await run(pause_job, job.id, project)
    assert paused.job.status == 'paused'

    # the message the start sent is still in the broker; the worker gets it and acks it
    runs: list[int] = []

    async def handler(ctx):
        runs.append(ctx.attempt)
        return {'ok': True}

    await asyncio.to_thread(run_job, kinds['t_pause'], handler, {}, str(job.id))
    assert runs == []
    assert sql(pg, 'SELECT status, attempt FROM jobs WHERE id = %s', job.id) == [('paused', 0)]

    resumed = await run(resume_job, job.id, project)
    assert resumed.job.status == 'queued'
    assert producer.sent() == [str(job.id), str(job.id)]  # dispatched again under the same id
    await asyncio.to_thread(run_job, kinds['t_pause'], handler, {}, str(job.id))
    assert runs == [1]
    assert sql(pg, 'SELECT status, retry_count FROM jobs WHERE id = %s', job.id) == [
        ('completed', 0)
    ]


@pytest.mark.asyncio
async def test_a_running_job_pauses_at_its_boundary_with_the_requested_reason(
    pg, job_env, producer, wakes, kinds
) -> None:
    import threading

    from src.tasks.runtime.context import Paused

    project = make_project(pg)
    job = (await run(start_job, 't_pause', project)).job
    entered = threading.Event()

    async def handler(ctx):
        entered.set()
        while not ctx.paused():
            await ctx.progress(0, None, 'working', force=True)
            await asyncio.sleep(0.05)
        return Paused()

    worker = threading.Thread(
        target=run_job, args=(kinds['t_pause'], handler, {}, str(job.id)), daemon=True
    )
    worker.start()
    assert await asyncio.to_thread(entered.wait, 20)
    asked = await run(pause_job, job.id, project)
    assert asked.job.status == 'running'
    await asyncio.to_thread(worker.join, 20)
    assert sql(
        pg, 'SELECT status, pause_reason, retry_count, owner FROM jobs WHERE id = %s', job.id
    ) == [('paused', 'user', 0, None)]
    assert sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id) == [('paused',)]


@pytest.mark.asyncio
async def test_a_handler_can_request_its_own_pause(pg, job_env, producer, wakes, kinds) -> None:
    from src.tasks.runtime.context import Paused

    project = make_project(pg)
    job = (await run(start_job, 't_pause', project)).job

    async def handler(ctx):
        await ctx.request_pause('budget')
        assert ctx.paused()
        return Paused()

    await asyncio.to_thread(run_job, kinds['t_pause'], handler, {}, str(job.id))
    assert sql(pg, 'SELECT status, pause_reason FROM jobs WHERE id = %s', job.id) == [
        ('paused', 'budget')
    ]

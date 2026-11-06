"""Cancel, pause and resume are database writes: no Celery revoke, no control command (D-71).

A fake producer and a fake worker control record every call. A stale delivery of a cancelled or
paused job is acknowledged and does nothing, and a pause followed by a resume still runs.
"""

from __future__ import annotations

import asyncio

import pytest
from celery.app.control import Control
from fixtures.job_rows import make_project, sql

from src.db.session import async_session_factory
from src.services.jobs.service import cancel_job, pause_job, resume_job, start_job
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db


@pytest.fixture
def control_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Any worker control command (revoke, terminate, broadcast, ...) is recorded and refused."""
    calls: list[str] = []

    def refuse(name: str):
        def trap(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f'Celery control command {name} was called')

        return trap

    for name in ('revoke', 'terminate', 'broadcast', 'revoke_by_stamped_headers'):
        monkeypatch.setattr(Control, name, refuse(name))
    return calls


async def run(fn, *args, **kwargs):
    async with async_session_factory()() as session:
        return await fn(session, *args, **kwargs)


def deliver(kinds, kind, job_id, handler):
    return asyncio.to_thread(run_job, kinds[kind], handler, {}, str(job_id))


@pytest.mark.asyncio
async def test_cancel_and_pause_send_no_control_command(
    pg, job_env, producer, wakes, kinds, control_calls
) -> None:
    project = make_project(pg)
    queued = (await run(start_job, 't_pause', project)).job
    await run(pause_job, queued.id, project)
    await run(cancel_job, queued.id, project, requested_by='tester')
    other = (await run(start_job, 't_pause', project)).job
    await run(cancel_job, other.id, project, requested_by='tester')
    assert control_calls == []
    assert {call for call, _ in producer.calls} <= {'send', 'mark', 'clear'}


@pytest.mark.asyncio
async def test_a_stale_delivery_of_a_cancelled_or_paused_job_is_acked_and_does_nothing(
    pg, job_env, producer, wakes, kinds, control_calls
) -> None:
    project = make_project(pg)
    runs: list[str] = []

    async def handler(ctx):
        runs.append(str(ctx.job_id))

    cancelled = (await run(start_job, 't_pause', project)).job
    await run(cancel_job, cancelled.id, project, requested_by='tester')
    paused = (await run(start_job, 't_pause', project)).job
    await run(pause_job, paused.id, project)
    for job in (cancelled, paused):
        await deliver(kinds, 't_pause', job.id, handler)  # returns: the message is acknowledged
    assert runs == []
    states = sql(
        pg,
        'SELECT status, attempt FROM jobs WHERE id = ANY(%s) ORDER BY status',
        [cancelled.id, paused.id],
    )
    assert states == [('cancelled', 0), ('paused', 0)]
    assert control_calls == []


@pytest.mark.asyncio
async def test_pause_then_resume_still_runs(
    pg, job_env, producer, wakes, kinds, control_calls
) -> None:
    project = make_project(pg)
    job = (await run(start_job, 't_pause', project)).job
    await run(pause_job, job.id, project)
    await run(resume_job, job.id, project)
    ran: list[int] = []

    async def handler(ctx):
        ran.append(ctx.attempt)
        return {'done': True}

    # the old message and the resume's message both arrive; the first claims, the second is a no-op
    await deliver(kinds, 't_pause', job.id, handler)
    await deliver(kinds, 't_pause', job.id, handler)
    assert ran == [1]
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('completed',)]
    assert control_calls == []
    assert producer.sent() == [str(job.id), str(job.id)]

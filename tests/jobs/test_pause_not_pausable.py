"""Only kinds marked ``pausable`` pause (02b section 2.1): the rest answer 409 job_not_pausable."""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, sql

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.jobs.service import pause_job, start_job
from src.tasks.runtime.context import Paused
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db


async def run(fn, *args, **kwargs):
    async with async_session_factory()() as session:
        return await fn(session, *args, **kwargs)


@pytest.mark.asyncio
async def test_a_kind_that_is_not_pausable_answers_409(pg, job_env, producer, wakes, kinds) -> None:
    project = make_project(pg)
    job = (await run(start_job, 't_plain', project)).job
    with pytest.raises(RequestRejectedError) as refused:
        await run(pause_job, job.id, project)
    assert (refused.value.status_code, refused.value.detail['error']) == (409, 'job_not_pausable')
    assert sql(pg, 'SELECT status, pause_requested_at IS NULL FROM jobs WHERE id = %s', job.id) == [
        ('queued', True)
    ]


@pytest.mark.asyncio
async def test_a_handler_of_a_kind_that_is_not_pausable_cannot_pause_itself(
    pg, job_env, producer, wakes, kinds
) -> None:
    import asyncio

    project = make_project(pg)
    job = (await run(start_job, 't_plain', project)).job

    async def handler(ctx):
        return Paused('audit')

    await asyncio.to_thread(run_job, kinds['t_plain'], handler, {}, str(job.id))
    assert sql(pg, 'SELECT status, error_code FROM jobs WHERE id = %s', job.id) == [
        ('failed', 'job_not_pausable')
    ]

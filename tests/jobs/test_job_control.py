"""cancel_job, pause_job and resume_job are database writes; a resume dispatches the same id."""

from __future__ import annotations

import uuid

import pytest
from fixtures.job_rows import make_project, sql

from src.core.exceptions import RequestRejectedError
from src.db.repos import jobs as repo
from src.db.session import async_session_factory, project_tx
from src.services.jobs.service import cancel_job, pause_job, resume_job, start_job


pytestmark = pytest.mark.db


async def run(fn, *args, **kwargs):
    async with async_session_factory()() as session:
        return await fn(session, *args, **kwargs)


async def start(project, kind='t_plain', **kwargs):
    return (await run(start_job, kind, project, **kwargs)).job


async def claim(project, job_id):
    async with async_session_factory()() as session, project_tx(session, project):
        return await session.run_sync(lambda s: repo.claim(s, job_id, 'w:1', 120))


@pytest.mark.asyncio
async def test_cancel_a_queued_job_is_immediate_and_clears_the_marker(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await start(project)
    result = await run(cancel_job, job.id, project, requested_by='tester')
    assert (result.job.status, result.changed) == ('cancelled', True)
    assert ('clear', str(job.id)) in producer.calls
    # the stale message for the cancelled row claims nothing
    refused = await claim(project, job.id)
    assert refused == repo.ClaimRefused('not_queued', 'cancelled')


@pytest.mark.asyncio
async def test_cancel_a_running_job_asks_it_to_stop_and_a_repeat_is_a_no_op(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await start(project)
    await claim(project, job.id)
    first = await run(cancel_job, job.id, project, requested_by='a')
    assert (first.job.status, first.changed) == ('cancelling', True)
    again = await run(cancel_job, job.id, project, requested_by='b')
    assert (again.job.status, again.changed) == ('cancelling', False)


@pytest.mark.asyncio
async def test_cancel_a_finished_job_is_409_and_an_unknown_one_404(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await start(project)
    await run(cancel_job, job.id, project, requested_by='a')
    with pytest.raises(RequestRejectedError) as finished:
        await run(cancel_job, job.id, project, requested_by='a')
    assert (finished.value.status_code, finished.value.detail['error']) == (409, 'job_finished')
    with pytest.raises(RequestRejectedError) as missing:
        await run(cancel_job, uuid.uuid4(), project, requested_by='a')
    assert (missing.value.status_code, missing.value.detail['error']) == (404, 'job_not_found')


@pytest.mark.asyncio
async def test_another_projects_job_is_not_found(pg, job_env, producer, wakes) -> None:
    mine, other = make_project(pg), make_project(pg)
    job = await start(other)
    with pytest.raises(RequestRejectedError) as missing:
        await run(cancel_job, job.id, mine, requested_by='a')
    assert missing.value.status_code == 404


@pytest.mark.asyncio
async def test_a_queued_job_pauses_at_once_and_resumes_under_the_same_id(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await start(project, 't_pause')
    paused = await run(pause_job, job.id, project)
    assert (paused.job.status, paused.job.pause_reason, paused.changed) == ('paused', 'user', True)
    # the stale message of the paused row is refused at the claim
    assert await claim(project, job.id) == repo.ClaimRefused('not_queued', 'paused')
    resumed = await run(resume_job, job.id, project)
    assert resumed.job.status == 'queued'
    assert resumed.job.pause_reason is None
    assert producer.sent() == [str(job.id), str(job.id)]  # the same id, dispatched again
    claimed = await claim(project, job.id)
    assert isinstance(claimed, repo.Claimed)
    assert claimed.job.attempt == 1
    assert claimed.job.retry_count == 0


@pytest.mark.asyncio
async def test_pause_of_a_running_job_only_asks(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = await start(project, 't_pause')
    await claim(project, job.id)
    asked = await run(pause_job, job.id, project)
    assert (asked.job.status, asked.job.pause_requested_at is not None) == ('running', True)
    repeat = await run(pause_job, job.id, project)
    assert repeat.changed is False


@pytest.mark.asyncio
async def test_pause_of_a_kind_that_is_not_pausable_is_409(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = await start(project, 't_plain')
    with pytest.raises(RequestRejectedError) as refused:
        await run(pause_job, job.id, project)
    assert (refused.value.status_code, refused.value.detail['error']) == (409, 'job_not_pausable')
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('queued',)]


@pytest.mark.asyncio
async def test_pause_of_a_cancelling_job_is_409(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = await start(project, 't_pause')
    await claim(project, job.id)
    await run(cancel_job, job.id, project, requested_by='a')
    with pytest.raises(RequestRejectedError) as refused:
        await run(pause_job, job.id, project)
    assert refused.value.detail['error'] == 'job_cancelling'


@pytest.mark.asyncio
async def test_resume_is_blocked_while_the_project_is_paused(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = await start(project, 't_pause')
    await run(pause_job, job.id, project)
    sql(pg, "UPDATE projects SET paused_at = now(), paused_by = 'test' WHERE id = %s", project)
    with pytest.raises(RequestRejectedError) as refused:
        await run(resume_job, job.id, project)
    assert refused.value.detail['error'] == 'resume_blocked'
    assert refused.value.detail['reason'] == 'project_paused'


@pytest.mark.asyncio
async def test_resume_of_a_job_that_is_not_paused_is_refused(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = await start(project, 't_pause')
    with pytest.raises(RequestRejectedError) as refused:
        await run(resume_job, job.id, project)
    assert refused.value.status_code == 409


@pytest.mark.asyncio
async def test_resume_while_the_group_is_busy_is_409_job_busy(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    first = await start(project, 't_excl', params={'target': 'x'})
    # an interrupted job leaves the group (interrupted is not active); a new start takes it
    token = (await claim(project, first.id)).job.fencing_token
    async with async_session_factory()() as session, project_tx(session, project):
        await session.run_sync(lambda s: repo.interrupt(s, first.id, token))
    second = await start(project, 't_excl', params={'target': 'x'})
    with pytest.raises(RequestRejectedError) as refused:
        await run(resume_job, first.id, project)
    assert refused.value.detail['error'] == 'job_busy'
    assert refused.value.detail['active_job_id'] == str(second.id)


@pytest.mark.asyncio
async def test_resume_with_the_broker_down_leaves_the_row_queued_for_the_reaper(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await start(project, 't_pause')
    await run(pause_job, job.id, project)
    producer.fail_send = True
    resumed = await run(resume_job, job.id, project)
    assert resumed.job.status == 'queued'
    assert sql(pg, 'SELECT status, last_dispatched_at FROM jobs WHERE id = %s', job.id) == [
        ('queued', None)
    ]

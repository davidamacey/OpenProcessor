"""The task wrapper claims once, runs the handler and writes one fenced terminal state (3A-5).

Celery's testing guide unit-tests with mocks and runs the code under test directly; the wrapper
is a plain function (``run_job``), so these tests call it without ``task_always_eager``
(https://docs.celeryq.dev/en/stable/userguide/testing.html).
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from fixtures.job_rows import make_project, project_session, sql, start

from src.db.engine import get_sync_engine
from src.db.repos import jobs as repo
from src.tasks.runtime.context import JobResult
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db


@pytest.fixture
def engine(job_env, wakes):
    return get_sync_engine()


def make_job(pg, engine, kinds, kind='t_plain', **overrides):
    project = make_project(pg)
    job = start(engine, project, kind, **overrides)
    return project, job


def run(kinds, job, handler, kind='t_plain'):
    run_job(kinds[kind], handler, {}, str(job.id))


def row(pg, job_id):
    cols = 'status, attempt, retry_count, error_code, error_message, result, owner, fencing_token'
    [found] = sql(pg, f'SELECT {cols} FROM jobs WHERE id = %s', job_id)
    return found


async def noop(ctx):
    return None


def test_a_noop_handler_completes(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds)

    async def handler(ctx):
        await ctx.progress(1, 1, 'done')
        return {'n': 1}

    run(kinds, job, handler)
    status, attempt, retry, code, _msg, result, owner, token = row(pg, job.id)
    assert (status, attempt, retry, code, result, owner, token) == (
        'completed',
        1,
        0,
        None,
        {'n': 1},
        None,
        1,
    )
    [(outcome,)] = sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id)
    assert outcome == 'completed'
    states = sql(
        pg,
        "SELECT payload->>'status' FROM events WHERE job_id = %s AND type = 'job.state' ORDER BY id",
        job.id,
    )
    assert [s for (s,) in states] == ['running', 'completed']


def test_a_result_with_errors_is_completed_with_errors(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds)

    async def handler(ctx):
        return JobResult({'bad': 2}, with_errors=True)

    run(kinds, job, handler)
    assert row(pg, job.id)[0] == 'completed_with_errors'


def test_the_handler_runs_once_and_sees_its_job(pg, engine, kinds) -> None:
    project, job = make_job(pg, engine, kinds, params={'a': 1})
    seen = []

    async def handler(ctx):
        seen.append((ctx.job_id, ctx.kind, ctx.project_id, dict(ctx.params), ctx.attempt))

    run(kinds, job, handler)
    assert seen == [(job.id, 't_plain', project, {'a': 1}, 1)]


def test_a_second_delivery_while_running_is_an_ack_and_a_no_op(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds)
    entered, release = threading.Event(), threading.Event()
    runs: list[int] = []

    async def slow(ctx):
        runs.append(1)
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.05)

    first = threading.Thread(target=run, args=(kinds, job, slow))
    first.start()
    assert entered.wait(20)
    run(kinds, job, slow)  # the redelivery returns at once: acked, nothing runs
    assert runs == [1]
    assert row(pg, job.id)[0] == 'running'
    release.set()
    first.join(20)
    assert row(pg, job.id)[0] == 'completed'
    assert runs == [1]


def test_a_terminal_row_is_a_no_op(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds)
    run(kinds, job, noop)
    runs = []

    async def handler(ctx):
        runs.append(1)

    run(kinds, job, handler)
    assert runs == []
    assert row(pg, job.id)[0] == 'completed'
    assert row(pg, job.id)[1] == 1


def test_an_unknown_job_id_is_acked(pg, engine, kinds) -> None:
    run_job(kinds['t_plain'], noop, {}, str(uuid.uuid4()))


def test_a_spent_retry_budget_ends_failed_without_running(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds)
    sql(pg, 'UPDATE jobs SET retry_count = max_attempts WHERE id = %s', job.id)
    runs = []

    async def handler(ctx):
        runs.append(1)

    run(kinds, job, handler)
    assert runs == []
    status, attempt, *_rest = row(pg, job.id)
    assert (status, attempt) == ('failed', 0)
    assert row(pg, job.id)[3] == 'attempts_exhausted'


def test_an_exception_fails_the_job_with_its_message(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds)

    async def handler(ctx):
        raise ValueError('bad input 7')

    run(kinds, job, handler)
    status, _attempt, retry, code, message, *_ = row(pg, job.id)
    assert (status, code, retry) == ('failed', 'handler_error', 0)
    assert message == 'ValueError: bad input 7'
    assert sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id) == [('failed',)]


def test_a_cancel_request_ends_cancelled(pg, engine, kinds) -> None:
    project, job = make_job(pg, engine, kinds)
    started = threading.Event()

    async def handler(ctx):
        started.set()
        while not ctx.cancelled():
            await ctx.progress(0, None, 'working', force=True)
            await asyncio.sleep(0.05)
        return {'stopped': True}

    worker = threading.Thread(target=run, args=(kinds, job, handler))
    worker.start()
    assert started.wait(20)
    with project_session(engine, project) as session:
        repo.cancel(session, job.id, 'tester')
    worker.join(20)
    status, *_ = row(pg, job.id)
    assert status == 'cancelled'
    assert sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id) == [('cancelled',)]


def test_a_soft_time_limit_from_celery_fails_with_time_limit(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds)

    async def handler(ctx):
        raise SoftTimeLimitExceeded

    run(kinds, job, handler)
    status, _attempt, retry, code, *_ = row(pg, job.id)
    assert (status, code, retry) == ('failed', 'time_limit', 1)
    assert sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id) == [
        ('time_limit',)
    ]


def test_the_own_soft_timer_writes_time_limit_without_waiting_for_a_thread(
    pg, engine, kinds
) -> None:
    """Spike S1 item 7: with ``asyncio.to_thread`` the terminal write came 20 s late."""
    _, job = make_job(pg, engine, kinds, 't_short')
    stuck = 8.0

    async def handler(ctx):
        await asyncio.to_thread(time.sleep, stuck)

    started = time.monotonic()
    run(kinds, job, handler, 't_short')
    elapsed = time.monotonic() - started
    status, _a, _r, code, *_ = row(pg, job.id)
    assert (status, code) == ('failed', 'time_limit')
    assert (
        elapsed < stuck - 2
    )  # written at the soft limit (4 s less its lead), not when the thread ends


def test_a_worker_shutdown_writes_interrupted_and_reraises(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds, auto_resume=True)

    async def handler(ctx):
        raise SystemExit(0)

    with pytest.raises(SystemExit):
        run(kinds, job, handler)
    status, _attempt, retry, *_ = row(pg, job.id)
    assert (status, retry) == ('interrupted', 1)
    assert sql(pg, 'SELECT auto_resume, owner FROM jobs WHERE id = %s', job.id) == [(True, None)]
    assert sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id) == [
        ('interrupted',)
    ]


def test_a_lost_fence_writes_nothing(pg, engine, kinds) -> None:
    _, job = make_job(pg, engine, kinds)

    async def handler(ctx):
        # a reaper took the job over and another delivery claimed it: the token moved on
        sql(
            pg,
            'UPDATE jobs SET fencing_token = fencing_token + 5, owner = %s WHERE id = %s',
            'w:other',
            job.id,
        )
        return {'late': True}

    run(kinds, job, handler)
    status, _a, _r, code, _m, result, owner, token = row(pg, job.id)
    assert (status, code, result, owner, token) == ('running', None, None, 'w:other', 6)
    assert sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id) == [(None,)]


@pytest.mark.parametrize('project_status', ['archived', 'deleting'])
def test_a_job_of_an_unavailable_project_fails_without_running(
    pg, engine, kinds, project_status
) -> None:
    project, job = make_job(pg, engine, kinds)
    if project_status == 'deleting':
        sql(
            pg,
            "UPDATE projects SET status = 'deleting', pre_delete_status = 'active' WHERE id = %s",
            project,
        )
    else:
        sql(pg, "UPDATE projects SET status = 'archived' WHERE id = %s", project)
    runs = []

    async def handler(ctx):
        runs.append(1)

    run(kinds, job, handler)
    assert runs == []
    status, attempt, _r, code, *_ = row(pg, job.id)
    assert (status, attempt, code) == ('failed', 0, 'project_unavailable')


def test_a_deployment_job_without_a_project_runs(pg, engine, kinds) -> None:
    from fixtures.job_rows import deployment_session, new_job

    with deployment_session(engine) as session:
        job = repo.insert_queued(session, new_job(None, 't_dep', scope='x')).job

    async def handler(ctx):
        assert ctx.project_id is None
        await ctx.progress(1, 1)

    run(kinds, job, handler, 't_dep')
    assert row(pg, job.id)[0] == 'completed'
    [(project,)] = sql(pg, 'SELECT project_id FROM events WHERE job_id = %s LIMIT 1', job.id)
    assert project is None


def test_a_paused_row_is_not_claimed(pg, engine, kinds) -> None:
    project, job = make_job(pg, engine, kinds, 't_pause')
    with project_session(engine, project) as session:
        repo.pause(session, job.id, 'user')
    runs = []

    async def handler(ctx):
        runs.append(1)

    run(kinds, job, handler, 't_pause')
    assert runs == []
    assert row(pg, job.id)[0] == 'paused'


def test_pause_then_resume_still_runs(pg, engine, kinds) -> None:
    project, job = make_job(pg, engine, kinds, 't_pause')
    with project_session(engine, project) as session:
        repo.pause(session, job.id, 'user')
    with project_session(engine, project) as session:
        repo.resume(session, job.id)
    run(kinds, job, noop, 't_pause')
    assert row(pg, job.id)[0] == 'completed'

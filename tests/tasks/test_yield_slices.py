"""A sliced handler gives way at a boundary: ``queued``, outcome ``yielded``, dispatched at once.

Research gap G-2. A yield keeps the attempt budget, and a kind without ``slice_s`` runs to
completion in one delivery.
"""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, sql, start

from src.db.engine import get_sync_engine
from src.tasks.runtime.context import Yield
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db


@pytest.fixture
def engine(job_env, wakes):
    return get_sync_engine()


def status(pg, job_id):
    return sql(
        pg, 'SELECT status, attempt, retry_count, dispatch_count FROM jobs WHERE id = %s', job_id
    )[0]


def test_a_returned_yield_requeues_and_dispatches_at_once(pg, engine, kinds, queue_hooks) -> None:
    project = make_project(pg)
    job = start(engine, project, 't_slice', exclusive_group='slicing')

    async def handler(ctx):
        await ctx.progress(1, 10, 'nodes')
        return Yield()

    run_job(kinds['t_slice'], handler, {}, str(job.id))
    assert status(pg, job.id) == ('queued', 1, 0, 1)  # one dispatch, no retry spent
    assert queue_hooks.calls[-2:] == [
        ('send', f'op.test.slice:{job.id}'),
        ('mark', str(job.id)),
    ]
    [(outcome, ended)] = sql(
        pg, 'SELECT outcome, ended_at IS NOT NULL FROM job_attempts WHERE job_id = %s', job.id
    )
    assert (outcome, ended) == ('yielded', True)
    [(state,)] = sql(
        pg,
        "SELECT payload->>'status' FROM events WHERE job_id = %s AND type = 'job.state' "
        'ORDER BY id DESC LIMIT 1',
        job.id,
    )
    assert state == 'queued'


def test_the_yielded_job_is_claimed_again_and_finishes(pg, engine, kinds, queue_hooks) -> None:
    project = make_project(pg)
    job = start(engine, project, 't_slice', exclusive_group='slicing')
    deliveries: list[int] = []

    async def handler(ctx):
        deliveries.append(ctx.attempt)
        if ctx.attempt < 3:
            return Yield()
        return {'slices': ctx.attempt}

    for _ in range(3):
        run_job(kinds['t_slice'], handler, {}, str(job.id))
    assert deliveries == [1, 2, 3]
    assert status(pg, job.id)[:3] == ('completed', 3, 0)
    outcomes = sql(
        pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s ORDER BY attempt', job.id
    )
    assert outcomes == [('yielded',), ('yielded',), ('completed',)]


def test_a_failed_send_leaves_the_row_queued_for_reaper_rule_c(
    pg, engine, kinds, queue_hooks
) -> None:
    project = make_project(pg)
    job = start(engine, project, 't_slice', exclusive_group='slicing')
    queue_hooks.fail = True

    async def handler(ctx):
        return Yield()

    run_job(kinds['t_slice'], handler, {}, str(job.id))
    assert status(pg, job.id) == ('queued', 1, 0, 0)
    assert sql(pg, 'SELECT last_dispatched_at FROM jobs WHERE id = %s', job.id) == [(None,)]


def test_a_kind_without_slice_s_cannot_yield(pg, engine, kinds, queue_hooks) -> None:
    project = make_project(pg)
    job = start(engine, project, 't_plain')

    async def handler(ctx):
        return Yield()

    run_job(kinds['t_plain'], handler, {}, str(job.id))
    state, _attempt, retry, _ = status(pg, job.id)
    assert (state, retry) == ('failed', 0)
    assert sql(pg, 'SELECT error_code FROM jobs WHERE id = %s', job.id) == [('yield_not_allowed',)]
    assert [c for c in queue_hooks.calls if c[0] == 'send'] == []


def test_a_kind_without_slice_s_runs_to_completion_in_one_delivery(
    pg, engine, kinds, queue_hooks
) -> None:
    project = make_project(pg)
    job = start(engine, project, 't_plain')
    units: list[int] = []

    async def handler(ctx):
        for unit in range(20):
            assert not ctx.slice_due()  # no slice_s: never due
            units.append(unit)
        return {'units': len(units)}

    run_job(kinds['t_plain'], handler, {}, str(job.id))
    assert status(pg, job.id)[:3] == ('completed', 1, 0)
    assert len(units) == 20

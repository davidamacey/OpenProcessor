"""Progress is written at most once per interval and on every stage change (trap T-31)."""

from __future__ import annotations

import asyncio

import pytest
from fixtures.job_rows import make_project, sql, start

from src.db.engine import get_sync_engine
from src.domain import job_timing
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db


@pytest.fixture
def job(pg, job_env, wakes, kinds):
    project = make_project(pg)
    return start(get_sync_engine(), project, 't_plain')


def progress_events(pg, job_id):
    rows = sql(
        pg,
        "SELECT payload FROM events WHERE job_id = %s AND type = 'job.progress' ORDER BY id",
        job_id,
    )
    return [payload['progress'] for (payload,) in rows]


def run(kinds, job, handler):
    run_job(kinds['t_plain'], handler, {}, str(job.id))


def test_a_tight_loop_in_one_stage_writes_once(pg, job, kinds, monkeypatch) -> None:
    monkeypatch.setattr(job_timing, 'PROGRESS_MIN_INTERVAL_S', 30)

    async def handler(ctx):
        for done in range(200):
            await ctx.progress(done, 200, 'scan')

    run(kinds, job, handler)
    events = progress_events(pg, job.id)
    first = events[0]
    assert first['done'] == 0
    # the wrapper writes the last skipped snapshot once at the end
    assert [e['done'] for e in events] == [0, 199]


def test_a_stage_change_always_writes(pg, job, kinds, monkeypatch) -> None:
    monkeypatch.setattr(job_timing, 'PROGRESS_MIN_INTERVAL_S', 30)

    async def handler(ctx):
        for stage in ('a', 'a', 'a', 'b', 'b', 'c'):
            await ctx.progress(1, 6, stage)

    run(kinds, job, handler)
    assert [e['stage'] for e in progress_events(pg, job.id)] == ['a', 'b', 'c']


def test_a_snapshot_is_the_full_state_not_a_delta(pg, job, kinds) -> None:
    async def handler(ctx):
        await ctx.progress(3, 10, 'embed', {'skipped': 2, 'failed': 1})

    run(kinds, job, handler)
    assert progress_events(pg, job.id) == [
        {'done': 3, 'total': 10, 'stage': 'embed', 'counters': {'skipped': 2, 'failed': 1}}
    ]
    assert sql(pg, 'SELECT progress FROM jobs WHERE id = %s', job.id) == [
        ({'done': 3, 'total': 10, 'stage': 'embed', 'counters': {'skipped': 2, 'failed': 1}},)
    ]


def test_the_last_skipped_snapshot_is_stored_when_the_job_ends(pg, job, kinds, monkeypatch) -> None:
    monkeypatch.setattr(job_timing, 'PROGRESS_MIN_INTERVAL_S', 30)

    async def handler(ctx):
        for done in range(1, 51):
            await ctx.progress(done, 50, 'scan')

    run(kinds, job, handler)
    [(stored,)] = sql(pg, 'SELECT progress FROM jobs WHERE id = %s', job.id)
    assert stored['done'] == 50


def test_a_later_call_after_the_interval_writes_again(pg, job, kinds, monkeypatch) -> None:
    monkeypatch.setattr(job_timing, 'PROGRESS_MIN_INTERVAL_S', 0.2)

    async def handler(ctx):
        await ctx.progress(1, 3, 'scan')
        await ctx.progress(2, 3, 'scan')  # inside the interval: kept, not written
        await asyncio.sleep(0.3)
        await ctx.progress(3, 3, 'scan')

    run(kinds, job, handler)
    assert [e['done'] for e in progress_events(pg, job.id)] == [1, 3]


def test_force_writes_inside_the_interval(pg, job, kinds, monkeypatch) -> None:
    monkeypatch.setattr(job_timing, 'PROGRESS_MIN_INTERVAL_S', 30)

    async def handler(ctx):
        await ctx.progress(1, 3, 'scan')
        await ctx.progress(2, 3, 'scan', force=True)

    run(kinds, job, handler)
    assert [e['done'] for e in progress_events(pg, job.id)] == [1, 2]

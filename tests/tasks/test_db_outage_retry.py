"""A handler's control-plane writes (progress, checkpoint, pause) outlast a database restart.

Plan 02b step 9: "the job completes or is resumed". Without the retry the first write after the
database went away raised OperationalError, the handler failed, the terminal write failed too and
the row stayed `running` until its lease expired.
"""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, sql, start
from sqlalchemy.exc import OperationalError
from tenacity import wait_none

from src.db.engine import get_sync_engine
from src.db.repos import jobs as repo
from src.tasks.runtime import context
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db


@pytest.fixture
def engine(job_env, wakes):
    return get_sync_engine()


def _outage(real, failures: int):
    calls = {'n': 0}

    def flaky(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] <= failures:
            raise OperationalError(
                'UPDATE jobs', {}, Exception('the database system is shutting down')
            )
        return real(*args, **kwargs)

    return flaky, calls


@pytest.mark.parametrize('write', ['checkpoint', 'progress'])
def test_a_write_waits_for_the_database_and_the_job_completes(
    pg, engine, kinds, queue_hooks, monkeypatch, write
) -> None:
    monkeypatch.setattr(context, 'DB_OUTAGE_WAIT', wait_none(), raising=False)
    flaky, calls = _outage(getattr(repo, write), failures=3)
    monkeypatch.setattr(context.repo, write, flaky)
    project = make_project(pg)
    job = start(engine, project, 't_plain')

    async def handler(ctx):
        if write == 'checkpoint':
            await ctx.checkpoint({'done': 1})
        else:
            await ctx.progress(1, 2, 'working', force=True)
        return {'ok': True}

    run_job(kinds['t_plain'], handler, {}, str(job.id))
    assert calls['n'] == 4  # three failures, then the write that landed
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('completed',)]


def test_an_outage_longer_than_the_budget_still_fails_the_write(
    pg, engine, kinds, queue_hooks, monkeypatch
) -> None:
    monkeypatch.setattr(context, 'DB_OUTAGE_WAIT', wait_none(), raising=False)
    monkeypatch.setattr(context, 'DB_OUTAGE_BUDGET_S', 0.05, raising=False)
    flaky, _calls = _outage(repo.checkpoint, failures=10**9)
    monkeypatch.setattr(context.repo, 'checkpoint', flaky)
    project = make_project(pg)
    job = start(engine, project, 't_plain')
    seen: list[BaseException] = []

    async def handler(ctx):
        try:
            await ctx.checkpoint({'done': 1})
        except OperationalError as exc:
            seen.append(exc)
            raise
        return {}

    run_job(kinds['t_plain'], handler, {}, str(job.id))
    assert len(seen) == 1

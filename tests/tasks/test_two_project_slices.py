"""One consumer, two projects' loops: slices let project B start within one slice of project A.

Research gap G-2: a loop that ran for hours held its queue slot, and the other project's loop
waited for the whole run. With ``Yield`` at a boundary after ``slice_s`` the queue interleaves
them. The consumer here is one thread popping a FIFO of delivered messages, the way a worker with
concurrency 1 reads one queue; a yield dispatches the same job id to the back of the FIFO.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import pytest
from fixtures.job_rows import make_project, sql, start

from src.db.engine import get_sync_engine
from src.tasks.runtime.context import Yield
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db

UNIT_S = 0.05
UNITS = 40  # 2 s of work: two slices of 1 s


def test_project_b_starts_within_one_slice_of_project_a(
    pg, job_env, wakes, kinds, queue_hooks
) -> None:
    engine = get_sync_engine()
    a, b = make_project(pg), make_project(pg)
    jobs = {
        'a': start(engine, a, 't_slice', exclusive_group='slicing'),
        'b': start(engine, b, 't_slice', exclusive_group='slicing'),
    }
    names = {job.id: name for name, job in jobs.items()}
    units_done = {'a': 0, 'b': 0}
    first_unit: dict[str, float] = {}
    fifo: deque[str] = deque([str(jobs['a'].id), str(jobs['b'].id)])

    async def handler(ctx):
        name = names[ctx.job_id]
        while units_done[name] < UNITS:
            first_unit.setdefault(name, time.monotonic())
            await asyncio.sleep(UNIT_S)
            units_done[name] += 1
            if ctx.slice_due() and units_done[name] < UNITS:
                return Yield()
        return {'units': units_done[name]}

    original = queue_hooks.enqueue

    def enqueue(task_name, *, job_id):
        original(task_name, job_id=job_id)
        fifo.append(job_id)  # the broker delivers it to the one consumer, behind the others

    queue_hooks.enqueue = enqueue  # type: ignore[method-assign]
    started = time.monotonic()
    while fifo:
        run_job(kinds['t_slice'], handler, {}, fifo.popleft())

    slice_s = kinds['t_slice'].slice_s
    assert (
        first_unit['b'] - first_unit['a'] <= slice_s + 0.5
    )  # B waited one slice, not A's whole run
    assert units_done == {'a': UNITS, 'b': UNITS}
    for job in jobs.values():
        [(state, attempts)] = sql(pg, 'SELECT status, attempt FROM jobs WHERE id = %s', job.id)
        assert state == 'completed'
        assert attempts >= 2  # each ran in more than one slice
    assert time.monotonic() - started < 30

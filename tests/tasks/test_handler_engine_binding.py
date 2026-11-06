"""A handler's services reach the database through the task's own engine (02b section 8).

The runtime builds one ``NullPool`` async engine per task. Every service the handler calls opens
its sessions through ``async_session_factory``, which must use that engine: a Celery child has
no ``WEB_CONCURRENCY``, so falling back to the API engine fails closed there.
"""

from __future__ import annotations

import asyncio

import pytest
from fixtures.job_rows import make_project, sql, start

from src.db import engine as db_engine
from src.services.events.publish import GLOBAL, apublish
from src.tasks.runtime.job_task import run_job


pytestmark = pytest.mark.db


@pytest.fixture
def engine(job_env, wakes):
    return db_engine.get_sync_engine()


def test_a_handler_and_a_task_it_spawns_publish_through_the_task_engine(pg, engine, kinds) -> None:
    job = start(engine, make_project(pg), 't_plain')
    used: list[object] = []

    async def spawned() -> None:
        used.append(db_engine.current_async_engine())
        await apublish('project.spawned_probe', {'n': 2}, project=GLOBAL)

    async def handler(ctx):
        used.append(db_engine.current_async_engine())
        await apublish('project.handler_probe', {'n': 1}, project=GLOBAL)
        await asyncio.create_task(spawned())

    run_job(kinds['t_plain'], handler, {}, str(job.id))

    [(status, code)] = sql(pg, 'SELECT status, error_code FROM jobs WHERE id = %s', job.id)
    assert (status, code) == ('completed', None)
    types = sql(
        pg,
        "SELECT type FROM events WHERE type IN ('project.handler_probe', 'project.spawned_probe') "
        'ORDER BY type',
    )
    assert types == [('project.handler_probe',), ('project.spawned_probe',)]
    assert len({id(e) for e in used}) == 1  # both saw the one task engine
    with pytest.raises(Exception, match='WEB_CONCURRENCY'):
        db_engine.current_async_engine()  # and the binding ended with the task

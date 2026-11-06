"""Trap T-1: every task builds its own NullPool async engine in its loop and disposes it first."""

from __future__ import annotations

import asyncio

import pytest
from fixtures.job_rows import make_project, start
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine

from src.db import engine as db_engine
from src.tasks.runtime import job_task


pytestmark = pytest.mark.db


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch, job_env, wakes):
    built: list[dict] = []
    real = db_engine.new_task_async_engine
    records: dict[int, dict] = {}
    original_dispose = AsyncEngine.dispose

    def recording(application_name=None):
        engine = real(application_name)
        records[id(engine)] = {
            'engine': engine,
            'disposed': False,
            'loop_open_at_dispose': None,
        }
        built.append(records[id(engine)])
        return engine

    async def dispose(self, *args, **kwargs):
        record = records.get(id(self))
        if record is not None:
            record['disposed'] = True
            record['loop_open_at_dispose'] = not asyncio.get_running_loop().is_closed()
        return await original_dispose(self, *args, **kwargs)

    monkeypatch.setattr(job_task, 'new_task_async_engine', recording)
    monkeypatch.setattr(AsyncEngine, 'dispose', dispose)
    return built


def test_each_task_gets_its_own_nullpool_engine_disposed_inside_its_loop(
    pg, engines, kinds
) -> None:
    project = make_project(pg)
    engine = db_engine.get_sync_engine()
    jobs = [start(engine, project, 't_plain') for _ in range(2)]

    async def handler(ctx):
        await ctx.progress(1, 1)  # uses the engine

    for job in jobs:
        job_task.run_job(kinds['t_plain'], handler, {}, str(job.id))

    assert len(engines) == 2
    assert engines[0]['engine'] is not engines[1]['engine']
    for record in engines:
        assert isinstance(record['engine'].sync_engine.pool, pool.NullPool)
        assert record['disposed']
        assert record['loop_open_at_dispose']  # disposed before the loop closed

"""``active_bakeoffs`` lists the bake-offs that hold the GPUs, across every project."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from _spec import spec_and_accepted
from fixtures.job_rows import deployment_session, make_project, sql

from src.db.repos import bakeoff as bakeoff_repo
from src.db.session import async_session_factory
from src.services.jobs.service import start_job
from src.services.training import gpu_arbiter


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


async def _start(pg, tmp_path) -> uuid.UUID:
    project = make_project(pg)
    spec, _ = spec_and_accepted(f'job{uuid.uuid4().hex[:8]}', str(tmp_path / 'out'))

    async def rows(session: Any, job: Any) -> None:
        await session.run_sync(
            lambda sync: bakeoff_repo.insert_run(
                sync,
                job_id=job.id,
                project_id=project,
                eval_dataset_keys=['export:e1'],
                models=['baseline:m1'],
            )
        )

    async with async_session_factory()() as session:
        started = await start_job(
            session, 'bakeoff', project, params=spec.model_dump(mode='json'), with_rows=rows
        )
    return started.job.id


def _active_ids(pg) -> set[uuid.UUID]:
    with deployment_session(pg.engine()) as session:
        return {row.job_id for row in bakeoff_repo.active_bakeoffs(session)}


@pytest.mark.asyncio
async def test_a_queued_bakeoff_of_any_project_holds_the_gpus(
    pg, job_env, producer, wakes, tmp_path
) -> None:
    job_id = await _start(pg, tmp_path)
    assert job_id in _active_ids(pg)
    assert await gpu_arbiter._bakeoff_holds_gpus() is True


@pytest.mark.asyncio
async def test_a_finished_bakeoff_releases_the_gpus(pg, job_env, producer, wakes, tmp_path) -> None:
    job_id = await _start(pg, tmp_path)
    assert job_id in _active_ids(pg)
    sql(pg, "UPDATE jobs SET status = 'cancelled', finished_at = now() WHERE id = %s", job_id)
    assert job_id not in _active_ids(pg)

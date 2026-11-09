"""A queued bake-off pins the run and the export it scores, read from PostgreSQL (no job file)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from _spec import spec_and_accepted
from fixtures.job_rows import make_project

from src.db.repos import bakeoff as bakeoff_repo
from src.db.session import async_session_factory
from src.services.curation.export_retention import collect_export_pins, plan_export_prune
from src.services.jobs.service import start_job
from src.services.training.run_retention import collect_active_bakeoff_run_ids, plan_run_prune


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


async def _start_bakeoff(pg: Any, tmp_path: Any, run_id: str, export_dir: str) -> None:
    project = make_project(pg)
    spec, _ = spec_and_accepted(f'job{uuid.uuid4().hex[:8]}', str(tmp_path / 'out'))
    spec.models[0].run_id = run_id
    spec.datasets[0].path = export_dir

    async def rows(session: Any, job: Any) -> None:
        await session.run_sync(
            lambda sync: bakeoff_repo.insert_run(
                sync,
                job_id=job.id,
                project_id=project,
                eval_dataset_keys=['export:e1'],
                models=['run:r'],
            )
        )

    async with async_session_factory()() as session:
        await start_job(
            session, 'bakeoff', project, params=spec.model_dump(mode='json'), with_rows=rows
        )


@pytest.mark.asyncio
async def test_a_queued_bakeoff_pins_its_run_and_its_export(
    pg, job_env, producer, wakes, tmp_path
) -> None:
    export_root = tmp_path / 'exports'
    scored = export_root / '20260101T000000Z'
    spare = export_root / '20260102T000000Z'
    for d in (scored, spare):
        d.mkdir(parents=True)
    run_id = '20260101T000000Z-run'
    await _start_bakeoff(pg, tmp_path, run_id, str(scored))

    assert await _in_thread(collect_active_bakeoff_run_ids) == {run_id}

    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    for name in (run_id, '20260102T000000Z-run'):
        (jobs_dir / f'{name}.status.json').write_text('{"state": "finished"}')
    pins = await _in_thread(lambda: collect_export_pins(export_root=export_root, jobs_dir=jobs_dir))
    assert scored.resolve() in pins
    assert plan_export_prune(export_root, 1, pins) == []
    runs = await _in_thread(collect_active_bakeoff_run_ids)
    assert (
        plan_run_prune(jobs_dir=jobs_dir, promoted_job_ids=set(), bakeoff_run_ids=runs, keep_last=1)
        == []
    )


async def _in_thread(fn: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(fn)

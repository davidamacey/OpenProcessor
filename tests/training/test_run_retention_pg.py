"""Run retention reads and deletes ``training_runs`` rows of the bound project, never a file."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fixtures.job_rows import sql

from src.db.repos import training_runs as runs_repo
from src.db.session import async_session_factory, project_tx
from src.services.curation import export_retention
from src.services.training import jobs
from src.services.training.job_models import TrainJobSpec
from src.services.training.run_retention import apply_run_prune, list_runs, plan_run_prune


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


async def _start(export_dir: str) -> uuid.UUID:
    spec = TrainJobSpec(dataset_export_dir=export_dir, cuda_visible_devices='0')
    async with async_session_factory()() as session:
        return uuid.UUID(await jobs.start_run(session, spec))


def _complete(pg, job_id: uuid.UUID) -> None:
    sql(
        pg,
        "UPDATE jobs SET status = 'running', owner = 'w', heartbeat_at = now(), "
        "lease_expires_at = now() + interval '1 hour', started_at = now() WHERE id = %s",
        job_id,
    )
    sql(
        pg,
        "UPDATE jobs SET status = 'completed', owner = NULL, lease_expires_at = NULL, "
        'finished_at = now() WHERE id = %s',
        job_id,
    )


@pytest.mark.asyncio
async def test_prune_removes_only_old_terminal_runs_and_their_rows(run_env) -> None:
    oldest, middle, newest = [await _start(f'/exports/e{i}') for i in range(3)]
    _complete(run_env.pg, oldest)
    _complete(run_env.pg, middle)  # newest stays queued: never pruned

    runs = await asyncio.to_thread(list_runs)
    assert [job_id for job_id, _ in runs] == [newest, middle, oldest]
    plan = plan_run_prune(runs, promoted_job_ids=set(), bakeoff_run_ids=set(), keep_last=1)
    assert plan == [middle, oldest]

    assert await asyncio.to_thread(apply_run_prune, plan) == {'removed_runs': 2}
    assert [job_id for job_id, _ in await asyncio.to_thread(list_runs)] == [newest]
    assert sql(run_env.pg, 'SELECT count(*) FROM jobs WHERE id = ANY(%s)', [middle, oldest]) == [
        (0,)
    ]


@pytest.mark.asyncio
async def test_a_runs_export_is_pinned_from_its_spec_and_its_manifest(run_env) -> None:
    await _start('/exports/queued')
    finished = await _start('/exports/original')
    manifest = {'lineage': {'export_dir': '/exports/repointed'}}
    async with async_session_factory()() as session, project_tx(session, run_env.project):
        await session.run_sync(lambda s: runs_repo.set_manifest(s, finished, manifest, None))

    pinned = await asyncio.to_thread(export_retention._run_export_dirs)
    assert pinned == {'/exports/queued', '/exports/original', '/exports/repointed'}

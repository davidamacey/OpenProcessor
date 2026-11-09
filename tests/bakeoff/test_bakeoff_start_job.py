"""``POST /bakeoff/run`` starts a ``bakeoff`` job with its ``bakeoff_runs`` row, never a job file."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from _spec import spec_and_accepted
from fixtures.job_rows import make_project, sql

from src.db.session import async_session_factory
from src.routers.curation import bakeoff as route
from src.schemas.bakeoff.run import BakeoffRunRequest


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture
def world(pg, job_env, producer, wakes, monkeypatch, tmp_path: Path) -> dict[str, Any]:
    project = make_project(pg)
    job_id = f'job{uuid.uuid4().hex[:8]}'
    spec, accepted = spec_and_accepted(job_id, str(tmp_path / 'out' / job_id))

    async def build(_request: Any, *, out_root: Path) -> Any:
        return spec, accepted

    monkeypatch.setattr(route.bakeoff_jobs, 'build_job_spec', build)
    monkeypatch.setattr(route, 'current_project_id', lambda: project, raising=False)
    monkeypatch.setattr(route, '_out_dir', lambda: tmp_path / 'out')
    return {'project': project, 'job_id': job_id, 'tmp': tmp_path, 'pg': pg, 'producer': producer}


async def run(payload: BakeoffRunRequest | None = None) -> Any:
    async with async_session_factory()() as session:
        return await route.bakeoff_run(payload or BakeoffRunRequest(), session)


@pytest.mark.asyncio
async def test_run_starts_a_bakeoff_job_with_its_row_and_writes_no_job_file(world) -> None:
    accepted = await run()
    assert accepted.job_id == world['job_id']
    [(kind, queue, project, params)] = sql(
        world['pg'],
        "SELECT kind, queue, project_id, params FROM jobs WHERE kind = 'bakeoff' "
        'AND project_id = %s',
        world['project'],
    )
    assert (kind, queue, project) == ('bakeoff', 'gpu_eval', world['project'])
    assert params['job_id'] == world['job_id']
    assert params['models'][0]['model'] == 'baseline:m1'
    [(models, datasets, results)] = sql(
        world['pg'],
        'SELECT models, eval_dataset_key, results_storage_key FROM bakeoff_runs '
        'WHERE project_id = %s',
        world['project'],
    )
    assert (models, datasets, results) == (['baseline:m1'], 'export:e1', None)
    assert len(world['producer'].sent()) == 1
    assert list(world['tmp'].rglob('*.job.json')) == []


@pytest.mark.asyncio
async def test_the_same_job_id_twice_is_a_conflict_and_starts_one_job(world) -> None:
    await run()
    with pytest.raises(Exception) as caught:  # noqa: PT011 - HTTPException, status checked below
        await run()
    assert getattr(caught.value, 'status_code', None) == 409
    assert sql(
        world['pg'],
        "SELECT count(*) FROM jobs WHERE kind = 'bakeoff' AND project_id = %s",
        world['project'],
    ) == [(1,)]

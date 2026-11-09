"""The ``bakeoff`` handler runs the evaluation once per job and records where the results are."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from _spec import spec_and_accepted
from fixtures.job_rows import make_project, sql

from src.db.session import async_session_factory
from src.services.jobs.service import start_job


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture
def consumer(queue_env, monkeypatch):
    from scripts.curation.bakeoff import bakeoff_tasks

    calls: list[dict[str, Any]] = []

    def fake(spec: dict[str, Any], cancelled: Any) -> dict[str, Any]:
        assert callable(cancelled)
        calls.append(spec)
        return {
            'state': 'done',
            'error': None,
            'progress': {'done': 1, 'total': 1},
            'completed': [{'dataset': 'export:e1', 'model': 'baseline:m1'}],
            'failed': [],
        }

    monkeypatch.setattr(bakeoff_tasks, 'evaluate', fake)
    bakeoff_tasks.app.loader.import_default_modules()
    return bakeoff_tasks, calls


async def start(pg, tmp_path):
    from src.db.repos import bakeoff as bakeoff_repo

    project = make_project(pg)
    spec, _ = spec_and_accepted(f'job{uuid.uuid4().hex[:8]}', str(tmp_path / 'out'))

    async def rows(session, job):
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
    return started.job.id, spec


async def deliver(tasks, job_id) -> Any:
    return await asyncio.to_thread(
        tasks.app.tasks['op.gpu_eval.bakeoff'].apply, args=[str(job_id)], task_id=str(job_id)
    )


@pytest.mark.asyncio
async def test_two_deliveries_of_one_job_run_one_evaluation(
    pg, job_env, producer, wakes, consumer, tmp_path
) -> None:
    tasks, calls = consumer
    job_id, spec = await start(pg, tmp_path)
    first = await deliver(tasks, job_id)
    second = await deliver(tasks, job_id)
    assert first.successful(), first.traceback
    assert second.successful(), second.traceback
    assert len(calls) == 1
    assert calls[0]['job_id'] == spec.job_id
    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', job_id)
    assert status == 'completed'
    assert result == {'out_dir': spec.out_dir, 'completed': 1, 'failed': 0}


@pytest.mark.asyncio
async def test_the_results_directory_is_recorded_on_the_run_row(
    pg, job_env, producer, wakes, consumer, tmp_path
) -> None:
    tasks, _ = consumer
    job_id, spec = await start(pg, tmp_path)
    await deliver(tasks, job_id)
    assert sql(pg, 'SELECT results_storage_key FROM bakeoff_runs WHERE job_id = %s', job_id) == [
        (spec.out_dir,)
    ]


@pytest.mark.asyncio
async def test_an_evaluation_that_ends_in_error_fails_the_job(
    pg, job_env, producer, wakes, consumer, tmp_path, monkeypatch
) -> None:
    tasks, _ = consumer
    monkeypatch.setattr(
        tasks,
        'evaluate',
        lambda _spec, _cancelled: {'state': 'error', 'error': 'no dataset could be scored'},
    )
    job_id, _ = await start(pg, tmp_path)
    await deliver(tasks, job_id)
    [(status, code, message)] = sql(
        pg, 'SELECT status, error_code, error_message FROM jobs WHERE id = %s', job_id
    )
    assert status == 'failed'
    assert 'no dataset could be scored' in message
    assert code


@pytest.mark.asyncio
async def test_a_cancelled_evaluation_ends_the_job_without_an_error(
    pg, job_env, producer, wakes, consumer, tmp_path, monkeypatch
) -> None:
    tasks, _ = consumer
    monkeypatch.setattr(tasks, 'evaluate', lambda _spec, _cancelled: {'state': 'cancelled'})
    job_id, spec = await start(pg, tmp_path)
    await deliver(tasks, job_id)
    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', job_id)
    assert status != 'failed'
    assert result == {'out_dir': spec.out_dir, 'cancelled': True}

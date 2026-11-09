"""The docs seeder fills a finished run's results into ``training_runs.manifest`` and stores the
confusion matrix under the run's artifact prefix."""

from __future__ import annotations

import uuid

import pytest

from scripts.curation import seed_docs_run_results as seeder
from src.db.repos import training_runs as runs_repo
from src.db.session import async_session_factory, project_tx
from src.domain.training_keys import run_object_key
from src.services.training import jobs
from src.services.training.job_models import TrainJobSpec


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


async def _run(run_env, manifest: dict | None) -> uuid.UUID:
    spec = TrainJobSpec(dataset_export_dir='/exports/e1', cuda_visible_devices='0')
    async with async_session_factory()() as session:
        job_id = uuid.UUID(await jobs.start_run(session, spec))
    if manifest is not None:
        async with async_session_factory()() as session, project_tx(session, run_env.project):
            await session.run_sync(
                lambda s: runs_repo.set_manifest(s, job_id, manifest, 'mlflow-1')
            )
    return job_id


@pytest.mark.asyncio
async def test_the_eval_lands_in_the_manifest_and_the_matrix_in_the_store(
    run_env, monkeypatch
) -> None:
    monkeypatch.setattr(seeder, 'build_rooted_store', lambda _roots: run_env.store)
    monkeypatch.setattr(seeder, 'storage_root_paths', dict)
    job_id = await _run(run_env, {'kind': 'train', 'results': {'final_state': 'finished'}})

    await seeder.apply(job_id, ['car', 'bus'], [10, 5])

    async with async_session_factory()() as session, project_tx(session, run_env.project):
        run = await session.run_sync(lambda s: runs_repo.get(s, job_id))
    assert run is not None
    assert run.mlflow_run_id == 'mlflow-1'
    results = run.manifest['results']
    assert results['final_state'] == 'finished'
    assert [row['name'] for row in results['eval']['per_class']] == ['car', 'bus']
    key = run_object_key(run.artifact_prefix, 'confusion_matrix.png')
    assert results['eval']['confusion_matrix_path'] == key
    assert run_env.store.exists(key)


@pytest.mark.asyncio
async def test_a_run_without_a_manifest_is_refused(run_env, monkeypatch) -> None:
    monkeypatch.setattr(seeder, 'build_rooted_store', lambda _roots: run_env.store)
    monkeypatch.setattr(seeder, 'storage_root_paths', dict)
    job_id = await _run(run_env, None)

    with pytest.raises(SystemExit, match='no manifest'):
        await seeder.apply(job_id, ['car', 'bus'], [3, 2])

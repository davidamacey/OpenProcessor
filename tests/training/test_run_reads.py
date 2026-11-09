"""A run's status, listing and cancel read the job row and the run row; no file is involved."""

from __future__ import annotations

import json
import uuid

import pytest
from fixtures.job_rows import sql

from src.db.repos import training_runs as runs_repo
from src.db.session import async_session_factory, project_tx
from src.services.training import job_campaigns, job_manifest, jobs
from src.services.training.job_models import CampaignRunSpec, TrainCampaignSpec, TrainJobSpec


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


async def _start(**overrides) -> str:
    spec = TrainJobSpec(dataset_export_dir='/exports/e1', cuda_visible_devices='0', **overrides)
    async with async_session_factory()() as session:
        return await jobs.start_run(session, spec)


def _run(pg, job_id: str, progress: dict | None = None) -> None:
    """Move the job to ``running`` the way a claim leaves it (owner, lease and heartbeat set)."""
    sql(
        pg,
        "UPDATE jobs SET status = 'running', owner = 'w', heartbeat_at = now(), "
        "lease_expires_at = now() + interval '1 hour', started_at = now(), "
        'progress = %s::jsonb WHERE id = %s',
        json.dumps(progress or {}),
        uuid.UUID(job_id),
    )


def _complete(pg, job_id: str) -> None:
    sql(
        pg,
        "UPDATE jobs SET status = 'completed', owner = NULL, lease_expires_at = NULL, "
        'finished_at = now() WHERE id = %s',
        uuid.UUID(job_id),
    )


@pytest.mark.asyncio
async def test_a_queued_run_reads_queued_and_a_running_one_reads_its_epoch(run_env) -> None:
    job_id = await _start()
    status = await jobs.read_status(job_id)
    assert (status.state, status.current_epoch) == ('queued', None)

    _run(
        run_env.pg,
        job_id,
        {
            'done': 4,
            'total': 10,
            'stage': 'running',
            'counters': {'last_epoch_metric': {'epoch': 4, 'map50': 0.4}},
        },
    )
    status = await jobs.read_status(job_id)
    assert (status.state, status.current_epoch, status.total_epochs) == ('running', 4, 10)
    assert status.last_epoch_metric == {'epoch': 4, 'map50': 0.4}
    active = await jobs.get_active_job()
    assert active is not None
    assert active.job_id == job_id


@pytest.mark.asyncio
async def test_a_finished_run_reads_its_manifest_results(run_env) -> None:
    job_id = await _start()
    manifest = {'results': {'eval': {'map50': 0.61}, 'checkpoint_path': '/runs/x/weights/best.pt'}}
    async with async_session_factory()() as session, project_tx(session, run_env.project):
        await session.run_sync(
            lambda s: runs_repo.set_manifest(s, uuid.UUID(job_id), manifest, 'mlflow-1')
        )
    _run(run_env.pg, job_id)
    _complete(run_env.pg, job_id)
    status = await jobs.read_status(job_id)
    assert status.state == 'finished'
    assert status.eval['map50'] == 0.61
    assert status.mlflow_run_id == 'mlflow-1'
    served = await job_manifest.read_manifest(job_id)
    assert served['results']['eval']['map50'] == 0.61


@pytest.mark.asyncio
async def test_listing_is_newest_first_and_scoped_to_the_project(run_env) -> None:
    first, second = await _start(), await _start()
    listed = [r.job_id for r in await jobs.list_runs(limit=10)]
    assert listed[:2] == [second, first]
    assert await jobs.count_runs() >= 2
    assert await jobs.read_status(str(uuid.uuid4())) is None


@pytest.mark.asyncio
async def test_a_bad_id_is_a_value_error(run_env) -> None:
    with pytest.raises(ValueError, match='invalid job_id'):
        await jobs.read_status('../../etc/passwd')


@pytest.mark.asyncio
async def test_cancel_ends_a_queued_run_at_once_and_an_unknown_run_is_not_found(run_env) -> None:
    job_id = await _start()
    assert await jobs.cancel_run(job_id) is True
    assert (await jobs.read_status(job_id)).state == 'cancelled'
    assert await jobs.cancel_run(str(uuid.uuid4())) is False


@pytest.mark.asyncio
async def test_the_run_log_tail_reads_the_stored_log(run_env) -> None:
    job_id = await _start()
    prefix = f'projects/{run_env.slug}/training/{job_id}'
    run_env.store.write_atomic(
        f'{prefix}/run.log', ''.join(f'line {i}\n' for i in range(50)).encode()
    )
    assert await jobs.tail_run_log(job_id, lines=3) == ['line 47', 'line 48', 'line 49']
    assert await jobs.tail_run_log(str(uuid.uuid4()), lines=3) == []


@pytest.mark.asyncio
async def test_a_stop_when_hit_skips_the_queued_siblings_and_the_wire_says_skipped(run_env) -> None:
    campaign = TrainCampaignSpec(
        dataset_export_dir='/exports/e1',
        cuda_visible_devices='0',
        runs=[CampaignRunSpec(profile='nano'), CampaignRunSpec(profile='small')],
    )
    async with async_session_factory()() as session:
        campaign_id, [first, second] = await job_campaigns.write_campaign(session, campaign)
    async with async_session_factory()() as session, project_tx(session, run_env.project):
        skipped = await session.run_sync(
            lambda s: runs_repo.skip_queued(
                s, campaign_id, uuid.UUID(first), 'auto_skip_threshold_met', 'stop_when met'
            )
        )
    assert skipped == 1
    assert (await jobs.read_status(second)).state == 'skipped'
    assert (await jobs.read_status(first)).state == 'queued'


@pytest.mark.asyncio
async def test_a_model_is_linked_to_the_run_it_came_from_once(run_env) -> None:
    job_id = await _start()
    model = uuid.uuid4()
    sql(
        run_env.pg,
        'INSERT INTO models (id, project_id, name, version, origin, family, task, labels, '
        "artifact_storage_key, created_by) VALUES (%s, %s, 'm', 1, 'trained', 'yolo26', "
        "'detect', ARRAY['a'], 'models/x', 'test')",
        model,
        run_env.project,
    )
    async with async_session_factory()() as session, project_tx(session, run_env.project):
        assert await session.run_sync(lambda s: runs_repo.link_model(s, model, uuid.UUID(job_id)))
        assert not await session.run_sync(
            lambda s: runs_repo.link_model(s, model, uuid.UUID(job_id))
        )
    [(linked,)] = sql(
        run_env.pg,
        'SELECT r.job_id FROM models m JOIN training_runs r ON r.id = m.training_run_id '
        'WHERE m.id = %s',
        model,
    )
    assert str(linked) == job_id


@pytest.mark.asyncio
async def test_the_status_never_carries_a_filesystem_path(run_env) -> None:
    job_id = await _start()
    manifest = {
        'results': {'eval': {'map50': 0.5, 'confusion_matrix_path': '/runs/x/confusion_matrix.png'}}
    }
    async with async_session_factory()() as session, project_tx(session, run_env.project):
        await session.run_sync(
            lambda s: runs_repo.set_manifest(s, uuid.UUID(job_id), manifest, None)
        )
    status = await jobs.read_status(job_id)
    assert 'confusion_matrix_path' not in status.eval
    assert status.eval['confusion_matrix_url'].endswith(
        f'/train/artifacts/{job_id}/confusion_matrix.png'
    )


@pytest.mark.asyncio
async def test_artifacts_are_served_from_the_run_prefix_and_only_whitelisted(run_env) -> None:
    job_id = await _start()
    prefix = f'projects/{run_env.slug}/training/{job_id}'
    run_env.store.write_atomic(f'{prefix}/confusion_matrix.png', b'png')
    run_env.store.write_atomic(f'{prefix}/best.pt', b'weights')
    served = await job_manifest.read_artifact(job_id, 'confusion_matrix.png')
    assert served is not None
    assert served.read_bytes() == b'png'
    assert await job_manifest.read_artifact(job_id, 'best.pt') is None  # not whitelisted
    assert await job_manifest.read_artifact(job_id, 'results.csv') is None  # never written
    assert await job_manifest.read_artifact(str(uuid.uuid4()), 'results.png') is None
    with pytest.raises(ValueError, match='invalid job_id'):
        await job_manifest.read_artifact('../x', 'results.png')


@pytest.mark.asyncio
async def test_promotion_is_stamped_on_the_stored_manifest(run_env) -> None:
    job_id = await _start()
    assert not await job_manifest.stamp_manifest_promotion(
        job_id, triton_name='m', promoted_at='t'
    )  # no manifest yet: nothing to stamp, not an error
    async with async_session_factory()() as session, project_tx(session, run_env.project):
        await session.run_sync(
            lambda s: runs_repo.set_manifest(s, uuid.UUID(job_id), {'kind': 'train'}, None)
        )
    assert await job_manifest.stamp_manifest_promotion(
        job_id, triton_name='m', promoted_at='t', force_used=True
    )
    manifest = await job_manifest.read_manifest(job_id)
    assert manifest['promoted_to']['triton_model_name'] == 'm'
    assert manifest['promoted_to']['force_used'] is True

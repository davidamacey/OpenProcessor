"""POST /train/start queues a ``train`` job and its ``training_runs`` row in one transaction."""

from __future__ import annotations

import uuid

import pytest
from fixtures.job_rows import sql

from src.config.task_routes import ROUTES
from src.db.session import async_session_factory
from src.services.training import job_campaigns, jobs
from src.services.training.job_models import CampaignRunSpec, TrainCampaignSpec, TrainJobSpec


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


def _spec(**overrides) -> TrainJobSpec:
    return TrainJobSpec(dataset_export_dir='/exports/e1', cuda_visible_devices='0', **overrides)


async def _start(spec: TrainJobSpec) -> str:
    async with async_session_factory()() as session:
        return await jobs.start_run(session, spec)


@pytest.mark.asyncio
async def test_a_start_writes_one_job_one_run_and_one_gpu_train_message(run_env) -> None:
    sent: list[tuple[str, str]] = []
    run_env.producer.on_send = lambda name, task_id: sent.append((name, task_id))

    job_id = await _start(_spec(include_classes=[2, 0]))

    [(kind, queue, status)] = sql(
        run_env.pg, 'SELECT kind, queue, status FROM jobs WHERE id = %s', uuid.UUID(job_id)
    )
    assert (kind, queue, status) == ('train', 'gpu_train', 'queued')
    [(family, names, prefix, spec_json)] = sql(
        run_env.pg,
        'SELECT family, class_names, artifact_prefix, spec FROM training_runs WHERE job_id = %s',
        uuid.UUID(job_id),
    )
    assert family == 'yolo26'
    assert names == ['car', 'person']  # the model's output order: ids 0 and 2, densely
    assert prefix == f'projects/{run_env.slug}/training/{job_id}'
    assert spec_json['dataset_sha'] == 'ds-sha'
    assert spec_json['registry_sha']
    assert spec_json['project'] == run_env.slug
    assert spec_json['mlflow_experiment'] == run_env.slug
    assert spec_json['project_export_root'].endswith('exports')
    assert spec_json['registry_snapshot_path'].endswith(f'{job_id}/registry_snapshot.json')
    assert sent == [('op.gpu_train.train', job_id)]
    assert ROUTES['op.gpu_train.train'] == 'gpu_train'


@pytest.mark.asyncio
async def test_a_start_creates_no_job_file(run_env, tmp_path) -> None:
    await _start(_spec())
    leftovers = [p.name for p in tmp_path.rglob('*') if p.name.endswith(('.job.json', '.cancel'))]
    assert leftovers == []


@pytest.mark.asyncio
async def test_single_cls_names_the_one_object_class(run_env) -> None:
    job_id = await _start(_spec(single_cls=True))
    [(names,)] = sql(
        run_env.pg, 'SELECT class_names FROM training_runs WHERE job_id = %s', uuid.UUID(job_id)
    )
    assert names == ['object']


@pytest.mark.asyncio
async def test_a_class_outside_the_registry_refuses_and_queues_nothing(run_env) -> None:
    with pytest.raises(ValueError, match='not in the class registry'):
        await _start(_spec(include_classes=[99]))
    assert sql(run_env.pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', run_env.project) == [
        (0,)
    ]


@pytest.mark.asyncio
async def test_a_campaign_queues_one_run_per_entry_in_order(run_env) -> None:
    campaign = TrainCampaignSpec(
        dataset_export_dir='/exports/e1',
        cuda_visible_devices='0',
        runs=[CampaignRunSpec(profile='nano', model_size='n'), CampaignRunSpec(profile='small')],
    )
    async with async_session_factory()() as session:
        campaign_id, job_ids = await job_campaigns.write_campaign(session, campaign)
    assert len(job_ids) == 2
    rows = sql(
        run_env.pg,
        "SELECT id, params->>'campaign_id', params->>'profile' FROM jobs "
        'WHERE project_id = %s ORDER BY created_at',
        run_env.project,
    )
    assert [(str(r[0]), r[1], r[2]) for r in rows] == [
        (job_ids[0], campaign_id, 'nano'),
        (job_ids[1], campaign_id, 'small'),
    ]
    async with async_session_factory()() as session:
        assert await job_campaigns.cancel_campaign(campaign_id) == 2
    statuses = sql(
        run_env.pg, 'SELECT status FROM jobs WHERE id = ANY(%s)', [uuid.UUID(j) for j in job_ids]
    )
    assert {r[0] for r in statuses} == {'cancelled'}

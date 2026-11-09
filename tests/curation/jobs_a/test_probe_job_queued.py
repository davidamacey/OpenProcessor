"""The probe job on the queue: start, busy, the run with its GPU claim, status and cancel (4A-3)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fixtures.job_rows import make_project, sql

from curation.jobs_a.ctx import FakeCtx
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.curation import probe_job
from src.services.jobs import status
from src.services.jobs.service import cancel_job


pytestmark = pytest.mark.db


async def start(project, **kw):
    args = {
        'train_job_id': 'run-1',
        'model_path': Path('/x/best.onnx'),
        'gpu': None,
        'architecture': 'yolo26',
        'resume': False,
    } | kw
    async with async_session_factory()() as session:
        return await probe_job.start(session, project, **args)


@pytest.mark.asyncio
async def test_start_queues_one_row_and_one_message_and_a_second_is_busy(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await start(project)
    assert (job.kind, job.status) == ('probe', 'queued')
    assert producer.sent() == [str(job.id)]
    with pytest.raises(RequestRejectedError) as busy:
        await start(project, train_job_id='run-2')
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')
    assert producer.sent() == [str(job.id)]


@pytest.mark.asyncio
async def test_an_unknown_architecture_is_refused_before_a_row_exists(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    with pytest.raises(ValueError, match='unknown probe architecture'):
        await start(project, architecture='nope')
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', project) == [(0,)]


@pytest.mark.asyncio
async def test_broker_down_is_503_and_the_row_failed(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    producer.fail_send = True
    with pytest.raises(RequestRejectedError) as down:
        await start(project)
    assert (down.value.status_code, down.value.detail['error']) == (503, 'queue_unavailable')
    assert sql(pg, 'SELECT status FROM jobs WHERE project_id = %s', project) == [('failed',)]


@pytest.mark.asyncio
async def test_cancel_while_queued_is_cancelled(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = await start(project)
    async with async_session_factory()() as session:
        moved = await cancel_job(session, job.id, project, requested_by='test')
    assert moved.job.status == 'cancelled'


@pytest.mark.asyncio
async def test_the_run_calls_the_inference_with_the_params_and_claims_the_gpu(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    project = make_project(pg)
    job = await start(project, gpu='0', resume=True)
    calls = []

    async def inference(model_path, opensearch, **kw):
        calls.append((model_path, kw))
        return 7

    claim, release = AsyncMock(return_value=['GPU-a']), AsyncMock()
    monkeypatch.setattr('src.services.curation.probe_predictions.run_probe_inference', inference)
    monkeypatch.setattr(probe_job, 'claim_cards', claim)
    monkeypatch.setattr(probe_job, 'release_cards', release)
    ctx = FakeCtx(project, job.id, job.params)

    result = await probe_job.run_probe(ctx, object())

    assert result == {'updated_count': 7, 'cancelled': False}
    [(path, kw)] = calls
    assert path == Path('/x/best.onnx')
    assert (kw['model_version'], kw['architecture'], kw['resume']) == ('run-1', 'yolo26', True)
    claim.assert_awaited_once()
    assert claim.await_args.args[0] is ctx
    release.assert_awaited_once_with(ctx)


@pytest.mark.asyncio
async def test_a_probe_without_a_gpu_claims_and_releases_nothing(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    project = make_project(pg)
    job = await start(project, gpu=None)

    async def inference(*_a, **_k):
        return 1

    claim, release = AsyncMock(), AsyncMock()
    monkeypatch.setattr('src.services.curation.probe_predictions.run_probe_inference', inference)
    monkeypatch.setattr(probe_job, 'claim_cards', claim)
    monkeypatch.setattr(probe_job, 'release_cards', release)

    await probe_job.run_probe(FakeCtx(project, job.id, job.params), object())

    claim.assert_not_called()
    release.assert_not_called()


@pytest.mark.asyncio
async def test_the_gpu_is_released_when_the_inference_fails(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    project = make_project(pg)
    job = await start(project, gpu='1')

    async def inference(*_a, **_k):
        raise RuntimeError('boom')

    release = AsyncMock()
    monkeypatch.setattr('src.services.curation.probe_predictions.run_probe_inference', inference)
    monkeypatch.setattr(probe_job, 'claim_cards', AsyncMock())
    monkeypatch.setattr(probe_job, 'release_cards', release)
    ctx = FakeCtx(project, job.id, job.params)
    with pytest.raises(RuntimeError, match='boom'):
        await probe_job.run_probe(ctx, object())
    release.assert_awaited_once_with(ctx)


@pytest.mark.asyncio
async def test_a_claim_the_arbiter_does_not_confirm_fails_the_run_before_any_inference(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    from src.tasks.runtime.gpu_claim import GpuBusyError

    project = make_project(pg)
    job = await start(project, gpu='0')
    inference, release = AsyncMock(), AsyncMock()
    monkeypatch.setattr('src.services.curation.probe_predictions.run_probe_inference', inference)
    monkeypatch.setattr(probe_job, 'claim_cards', AsyncMock(side_effect=GpuBusyError('gpu_busy')))
    monkeypatch.setattr(probe_job, 'release_cards', release)

    with pytest.raises(GpuBusyError):
        await probe_job.run_probe(FakeCtx(project, job.id, job.params), object())

    inference.assert_not_called()
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_claim_names_the_cards_at_the_container_positions_of_the_host_ids(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    from src.db.repos import gpu_devices
    from src.db.session import sync_deployment_tx
    from tests.training._cards import card

    project = make_project(pg)
    sql(pg, 'DELETE FROM gpu_reservations')
    sql(pg, 'DELETE FROM gpu_devices')
    # The trainer container is pinned to host cards 2,0: host card 0 is at position 1.
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '2,0')
    job = await start(project, gpu='0')
    claim = AsyncMock()

    async def inference(*_a, **_k):
        return 0

    monkeypatch.setattr('src.services.curation.probe_predictions.run_probe_inference', inference)
    monkeypatch.setattr(probe_job, 'claim_cards', claim)
    monkeypatch.setattr(probe_job, 'release_cards', AsyncMock())
    ctx = FakeCtx(project, job.id, job.params)
    await probe_job.run_probe(ctx, object())
    [find] = claim.await_args.args[1:]

    from sqlalchemy.orm import Session

    with Session(pg.engine('app')) as session, sync_deployment_tx(session, 'test'):
        gpu_devices.upsert(session, [card('GPU-a', 0), card('GPU-b', 1)])
    with Session(pg.engine('app')) as session, sync_deployment_tx(session, 'test'):
        assert [c.gpu_uuid for c in find(session)] == ['GPU-b']


@pytest.mark.asyncio
async def test_status_is_idle_without_a_job_then_the_newest_job(
    pg, job_env, producer, wakes, monkeypatch
) -> None:
    from src.services.curation.latest_job import latest_job

    project = make_project(pg)
    monkeypatch.setattr('src.services.curation.latest_job.current_project_id', lambda: project)
    async with async_session_factory()() as session:
        assert probe_job.status_dict(await latest_job(session, 'probe')) == {'status': 'idle'}
    job = await start(project)
    async with async_session_factory()() as session:
        shown = probe_job.status_dict(await latest_job(session, 'probe'))
        view = await status.get_job(session, job.id, project=project)
    assert shown['job_id'] == str(job.id) == view.id
    assert (shown['status'], shown['train_job_id'], shown['model_path']) == (
        'queued',
        'run-1',
        '/x/best.onnx',
    )

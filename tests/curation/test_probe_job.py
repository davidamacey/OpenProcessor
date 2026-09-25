"""C1: the in-process probe job wrapper (src.services.curation.probe_job)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.services.curation import probe_job


@pytest.fixture(autouse=True)
def _reset():
    probe_job._reset_for_tests()
    yield
    probe_job._reset_for_tests()


@pytest.mark.asyncio
async def test_start_probe_job_runs_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        return 42

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    status = await probe_job.start_probe_job(
        'job-1', 'train-1', Path('/tmp/best.onnx'), AsyncMock()
    )
    assert status['status'] == 'running'
    assert status['train_job_id'] == 'train-1'

    # Let the background task run to completion.
    for _ in range(50):
        await asyncio.sleep(0)
        if probe_job.get_status()['status'] != 'running':
            break
    final = probe_job.get_status()
    assert final['status'] == 'completed'
    assert final['updated_count'] == 42


@pytest.mark.asyncio
async def test_second_start_while_running_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        started.set()
        await release.wait()
        return 1

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    await probe_job.start_probe_job('job-1', 'train-1', Path('/tmp/best.onnx'), AsyncMock())
    await started.wait()
    with pytest.raises(probe_job.ProbeJobBusyError):
        await probe_job.start_probe_job('job-2', 'train-2', Path('/tmp/other.onnx'), AsyncMock())
    release.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if probe_job.get_status()['status'] != 'running':
            break


@pytest.mark.asyncio
async def test_failed_run_is_reported_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(model_path, opensearch, **kwargs):
        raise RuntimeError('onnx load failed')

    monkeypatch.setattr('src.services.curation.probe_predictions.run_probe_inference', _boom)
    await probe_job.start_probe_job('job-1', 'train-1', Path('/tmp/best.onnx'), AsyncMock())
    for _ in range(50):
        await asyncio.sleep(0)
        if probe_job.get_status()['status'] != 'running':
            break
    final = probe_job.get_status()
    assert final['status'] == 'failed'
    assert 'onnx load failed' in final['error']


@pytest.mark.asyncio
async def test_cancel_requests_task_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        started.set()
        await asyncio.sleep(10)
        return 1

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    await probe_job.start_probe_job('job-1', 'train-1', Path('/tmp/best.onnx'), AsyncMock())
    await started.wait()
    assert probe_job.cancel_probe_job() is True
    for _ in range(50):
        await asyncio.sleep(0)
        if probe_job.get_status()['status'] != 'running':
            break
    assert probe_job.get_status()['status'] == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_with_no_job_running_is_false() -> None:
    assert probe_job.cancel_probe_job() is False


@pytest.mark.asyncio
async def test_gpu_claim_failure_propagates_as_hard_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.training.gpu_arbiter import GpuArbiterStopFailedError

    async def _fail_claim(cuda_visible_devices):
        raise GpuArbiterStopFailedError('docker unavailable')

    monkeypatch.setattr('src.services.training.gpu_arbiter.claim_gpus_for_training', _fail_claim)
    with pytest.raises(GpuArbiterStopFailedError):
        await probe_job.start_probe_job(
            'job-1', 'train-1', Path('/tmp/best.onnx'), AsyncMock(), gpu='0'
        )
    assert probe_job.get_status()['status'] == 'idle'


@pytest.mark.asyncio
async def test_gpu_is_released_after_a_completed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    claimed = {'claim': False, 'release': False}

    async def _claim(cuda_visible_devices):
        claimed['claim'] = True

    async def _release(cuda_visible_devices):
        claimed['release'] = True

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        return 3

    monkeypatch.setattr('src.services.training.gpu_arbiter.claim_gpus_for_training', _claim)
    monkeypatch.setattr('src.services.training.gpu_arbiter.release_gpus_after_training', _release)
    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    await probe_job.start_probe_job(
        'job-1', 'train-1', Path('/tmp/best.onnx'), AsyncMock(), gpu='0'
    )
    for _ in range(50):
        await asyncio.sleep(0)
        if probe_job.get_status()['status'] != 'running':
            break
    assert claimed == {'claim': True, 'release': True}

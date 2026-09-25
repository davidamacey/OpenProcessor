"""C1: the file-backed probe job wrapper (src.services.curation.probe_job).

Multi-worker correctness is the point of this module (2026-09-25 fix):
``yolo-api`` runs under ``--workers=N`` -- separate OS processes that
don't share Python objects. Every test here points ``OP_PROBE_JOBS_DIR``
at a fresh ``tmp_path`` and, wherever a test claims to simulate "a second
process", it deliberately avoids relying on any module-level Python
object surviving between the two calls -- it only relies on the on-disk
state.json/heartbeat/cancel.flag/start.lock files, which is exactly what
a second real OS process would also only have access to. That is the
whole fix: after this rewrite there is no meaningful module-level state
left to desync across workers in the first place.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from src.services.curation import probe_job


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    d = tmp_path / 'probe'
    monkeypatch.setenv('OP_PROBE_JOBS_DIR', str(d))
    yield d
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

    for _ in range(50):
        await asyncio.sleep(0)
        if probe_job.get_status()['status'] != 'running':
            break
    final = probe_job.get_status()
    assert final['status'] == 'completed'
    assert final['updated_count'] == 42


# =============================================================================
# Cross-process visibility -- the actual bug
# =============================================================================


def test_status_read_by_a_second_process_sees_running(jobs_dir: Path) -> None:
    """Simulates the exact live failure: worker A's POST /probe/run writes
    state.json + heartbeat; worker B's GET /probe/status is a totally
    separate call into a module with an empty in-process history (no
    ``start_probe_job`` was ever called on this "process" -- we only
    write the files an OS process boundary would still leave behind).
    Before the fix this always read 'idle' because the state lived in a
    module-level dataclass instead of these files.
    """
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / 'state.json').write_text(
        json.dumps({'job_id': 'j-remote', 'status': 'running', 'train_job_id': 't1'})
    )
    (jobs_dir / 'heartbeat').touch()

    status = probe_job.get_status()
    assert status['status'] == 'running'
    assert status['job_id'] == 'j-remote'


@pytest.mark.asyncio
async def test_second_start_from_a_simulated_other_process_is_refused(jobs_dir: Path) -> None:
    """A job "started by another worker process" (state.json + fresh
    heartbeat on disk, no in-process task/asyncio.Lock involvement at
    all) must still refuse a start attempted from this process."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / 'state.json').write_text(
        json.dumps({'job_id': 'j-remote', 'status': 'running', 'train_job_id': 't1'})
    )
    (jobs_dir / 'heartbeat').touch()

    with pytest.raises(probe_job.ProbeJobBusyError):
        await probe_job.start_probe_job('job-2', 'train-2', Path('/tmp/other.onnx'), AsyncMock())


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
async def test_concurrent_starts_race_safely_only_one_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two `start_probe_job` calls launched genuinely concurrently (via
    ``asyncio.gather``, exercising the real ``fcntl.flock`` lock path
    rather than relying on cooperative ``await`` ordering) -- exactly one
    must succeed."""
    gate = asyncio.Event()

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        await gate.wait()
        return 1

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    results = await asyncio.gather(
        probe_job.start_probe_job('job-a', 'train-a', Path('/tmp/a.onnx'), AsyncMock()),
        probe_job.start_probe_job('job-b', 'train-b', Path('/tmp/b.onnx'), AsyncMock()),
        return_exceptions=True,
    )
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, probe_job.ProbeJobBusyError)]
    assert len(successes) == 1
    assert len(failures) == 1
    gate.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if probe_job.get_status()['status'] != 'running':
            break


# =============================================================================
# Stale-heartbeat repair
# =============================================================================


def test_stale_heartbeat_is_repaired_to_failed_on_status_read(jobs_dir: Path) -> None:
    """A worker process that owned the run died (or was killed) without
    ever writing a terminal status -- the heartbeat stops ticking. The
    *next* status poll, from any process, must notice and repair it
    rather than reporting 'running' forever."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / 'state.json').write_text(
        json.dumps({'job_id': 'j-dead', 'status': 'running', 'train_job_id': 't1'})
    )
    heartbeat = jobs_dir / 'heartbeat'
    heartbeat.touch()
    old = time.time() - (probe_job._HEARTBEAT_STALE_S + 30)
    os.utime(heartbeat, (old, old))

    status = probe_job.get_status()
    assert status['status'] == 'failed'
    assert 'stale' in status['error']
    on_disk = json.loads((jobs_dir / 'state.json').read_text())
    assert on_disk['status'] == 'failed'


@pytest.mark.asyncio
async def test_start_succeeds_after_stale_heartbeat_repair(
    jobs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a dead run's heartbeat is stale, a fresh start must be
    allowed -- the singleton lock must not treat a truly-dead run as
    still busy forever."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / 'state.json').write_text(
        json.dumps({'job_id': 'j-dead', 'status': 'running', 'train_job_id': 't1'})
    )
    heartbeat = jobs_dir / 'heartbeat'
    heartbeat.touch()
    old = time.time() - (probe_job._HEARTBEAT_STALE_S + 30)
    os.utime(heartbeat, (old, old))

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        return 7

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    status = await probe_job.start_probe_job(
        'job-fresh', 'train-fresh', Path('/tmp/fresh.onnx'), AsyncMock()
    )
    assert status['status'] == 'running'
    assert status['job_id'] == 'job-fresh'


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


# =============================================================================
# Cancel via the flag
# =============================================================================


@pytest.mark.asyncio
async def test_cancel_via_flag_is_observed_at_the_next_page_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_probe_inference`` (real code) checks its ``should_cancel``
    callable once per scroll page; ``cancel_probe_job`` only ever touches
    the cancel.flag file (never a direct ``asyncio.Task.cancel()``), so
    this is a genuine test of the cooperative-cancel wiring, not a
    hard-cancel shortcut."""
    started = asyncio.Event()
    page_count = {'n': 0}

    async def _fake_run_probe_inference(model_path, opensearch, *, should_cancel=None, **kwargs):
        started.set()
        while True:
            page_count['n'] += 1
            await asyncio.sleep(0)
            if should_cancel is not None and should_cancel():
                return page_count['n']

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    await probe_job.start_probe_job('job-1', 'train-1', Path('/tmp/best.onnx'), AsyncMock())
    await started.wait()
    assert probe_job.is_cancelled() is False
    assert probe_job.cancel_probe_job() is True
    assert probe_job.is_cancelled() is True
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
    # gpu release happens in _run's `finally`, after the ticker task is
    # cancelled/awaited -- that's a couple more await points past the
    # state.json write flipping to 'completed', so poll for the release
    # itself rather than stopping as soon as the status looks terminal.
    for _ in range(200):
        await asyncio.sleep(0)
        if claimed['release']:
            break
    assert claimed == {'claim': True, 'release': True}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

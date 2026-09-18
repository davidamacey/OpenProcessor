"""Tests for src/services/training/gpu_arbiter.py.

Ported from a private reference vehicle/license-plate curation stack's
training-pipeline test suite (Chunk 6). Covers:

* parse_cuda_visible_devices edge cases
* needs_multi_gpu_stop dispatch (single vs multi GPU)
* sentinel set / clear idempotence
* multi-GPU stop path falls back to sentinel when docker is missing
* reconcile leaves an active job intact, but cleans up when no jobs are active

The reference tests monkeypatched module-level ``DEFAULT_SENTINEL_PATH`` /
``TRAINING_LOCK_PATH`` constants. This port derives both from
``CurationConfig.state_dir`` instead (there is no fixed deployment-specific
path in the generic module), so the equivalent redirection here
monkeypatches ``gpu_arbiter._state_dir``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.services.training import gpu_arbiter as ga


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# parse_cuda_visible_devices
# ---------------------------------------------------------------------------


def test_parse_none_returns_empty():
    assert ga.parse_cuda_visible_devices(None) == []


def test_parse_empty_string_returns_empty():
    assert ga.parse_cuda_visible_devices('') == []


def test_parse_single_device():
    assert ga.parse_cuda_visible_devices('0') == [0]


def test_parse_dual_device_with_whitespace():
    assert ga.parse_cuda_visible_devices('  0 ,  2 ') == [0, 2]


def test_parse_invalid_token_skipped(caplog):
    # 'a' is unparseable; the integer survives.
    assert ga.parse_cuda_visible_devices('0,a,2') == [0, 2]


# ---------------------------------------------------------------------------
# needs_multi_gpu_stop
# ---------------------------------------------------------------------------


def test_needs_multi_gpu_stop_single_gpu_false():
    assert ga.needs_multi_gpu_stop('0') is False


def test_needs_multi_gpu_stop_dual_gpu_true():
    assert ga.needs_multi_gpu_stop('0,2') is True


def test_needs_multi_gpu_stop_none_false():
    assert ga.needs_multi_gpu_stop(None) is False


def test_needs_gemma_stop_alias_matches():
    """Back-compat alias: needs_gemma_stop == needs_multi_gpu_stop."""
    assert ga.needs_gemma_stop is ga.needs_multi_gpu_stop


# NOTE: the reference test suite had a `test_needs_gemma_stop_single_gpu2_true`
# pinning a deployment-specific fact -- "GPU 2 always hosts the Gemma vLLM
# server, so a lone '2' claim must still stop it." That knowledge doesn't
# exist in the generic module (no GPU has a fixed role here); a lone-GPU
# claim never requires a multi-GPU stop regardless of which id it names.
# GpuArbiterConfig.allowed_gpu_ids / .containers are the generic
# replacement for "which GPU/containers matter", see test_gpu_arbiter_config.py.


# ---------------------------------------------------------------------------
# sentinel set / clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_creates_sentinel(tmp_path: Path):
    sentinel = tmp_path / 'pause.sentinel'
    res = await ga.pause_gpu_worker(sentinel=sentinel)
    assert sentinel.exists()
    assert res.action == 'sentinel_set'


@pytest.mark.asyncio
async def test_pause_is_idempotent(tmp_path: Path):
    sentinel = tmp_path / 'pause.sentinel'
    await ga.pause_gpu_worker(sentinel=sentinel)
    res = await ga.pause_gpu_worker(sentinel=sentinel)
    assert res.action == 'noop'
    assert sentinel.exists()


@pytest.mark.asyncio
async def test_resume_removes_sentinel(tmp_path: Path):
    sentinel = tmp_path / 'pause.sentinel'
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    res = await ga.resume_gpu_worker(sentinel=sentinel)
    assert not sentinel.exists()
    assert res.action == 'sentinel_cleared'


@pytest.mark.asyncio
async def test_resume_when_absent_is_noop(tmp_path: Path):
    sentinel = tmp_path / 'pause.sentinel'
    res = await ga.resume_gpu_worker(sentinel=sentinel)
    assert res.action == 'noop'


# ---------------------------------------------------------------------------
# top-level claim/release dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_single_gpu_uses_sentinel(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    res = await ga.claim_gpus_for_training('0')
    assert res.action == 'sentinel_set'
    assert ga.sentinel_path().exists()


@pytest.mark.asyncio
async def test_claim_dual_gpu_falls_back_to_sentinel_without_docker(tmp_path: Path, monkeypatch):
    """Dual-GPU on a host without docker on PATH = sentinel-only fallback,
    but only once at least one container is configured -- otherwise
    stop_gpu_services() short-circuits to a pure no-op before ever
    checking docker (see test_gpu_arbiter_config.py)."""
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    monkeypatch.setattr(ga, '_docker_client', lambda: None)

    from src.config import GpuArbiterConfig, gpu_arbiter as gpu_arbiter_config_module

    fake_cfg = GpuArbiterConfig(containers=('fake-gpu-service',))
    monkeypatch.setattr(gpu_arbiter_config_module, '_default_gpu_arbiter_config', fake_cfg)

    res = await ga.claim_gpus_for_training('0,2')
    assert res.action == 'sentinel_set'
    assert ga.sentinel_path().exists()


@pytest.mark.asyncio
async def test_release_single_gpu_clears_sentinel(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    ga.sentinel_path().parent.mkdir(parents=True, exist_ok=True)
    ga.sentinel_path().touch()
    res = await ga.release_gpus_after_training('0')
    assert res.action == 'sentinel_cleared'
    assert not ga.sentinel_path().exists()


# ---------------------------------------------------------------------------
# reconcile_on_startup
# ---------------------------------------------------------------------------


def _write_status(jobs_dir: Path, job_id: str, state: str) -> None:
    payload = {'job_id': job_id, 'state': state}
    (jobs_dir / f'{job_id}.status.json').write_text(json.dumps(payload))


def _write_job(jobs_dir: Path, job_id: str, cuda_visible_devices: str | None = None) -> None:
    payload = {'job_id': job_id, 'cuda_visible_devices': cuda_visible_devices}
    (jobs_dir / f'{job_id}.job.json').write_text(json.dumps(payload))


@pytest.mark.asyncio
async def test_reconcile_skips_when_active_job_present(tmp_path: Path, monkeypatch):
    """An active run mid-session means: do nothing on API reboot."""
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    sentinel = tmp_path / 'pause.sentinel'
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()  # an active run set this

    # reconcile_on_startup keys "active" off *.job.json (written by the API
    # at submit time); *.status.json alone is not enough.
    # cuda_visible_devices='0' (single-GPU) routes into the pause-only
    # branch, matching this test's "sentinel stays untouched" assertion.
    _write_job(jobs_dir, 'job-a', cuda_visible_devices='0')
    _write_status(jobs_dir, 'job-a', 'running')

    res = await ga.reconcile_on_startup(train_jobs_dir=jobs_dir, sentinel=sentinel)
    assert res.action == 'noop'
    # Sentinel must remain -- we don't want to wake the worker while the
    # trainer is still running.
    assert sentinel.exists()


@pytest.mark.asyncio
async def test_reconcile_clears_sentinel_when_no_active_jobs(tmp_path: Path, monkeypatch):
    """No active run = trainer crashed; clear stale state.

    Under the default (unconfigured) GpuArbiterConfig there are no
    containers to restart, so the final reported action is the
    ``start_gpu_services()`` no-op -- but the sentinel clear (from the
    ``resume_gpu_worker`` call earlier in the same branch) still happens.
    """
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    sentinel = tmp_path / 'pause.sentinel'
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()

    _write_status(jobs_dir, 'job-a', 'failed')
    _write_status(jobs_dir, 'job-b', 'finished')

    monkeypatch.setattr(ga, '_docker_client', lambda: None)

    res = await ga.reconcile_on_startup(train_jobs_dir=jobs_dir, sentinel=sentinel)
    assert res.action == 'noop'
    assert not sentinel.exists()


@pytest.mark.asyncio
async def test_reconcile_handles_missing_jobs_dir(tmp_path: Path, monkeypatch):
    jobs_dir = tmp_path / 'no-such-dir'
    sentinel = tmp_path / 'pause.sentinel'

    monkeypatch.setattr(ga, '_docker_client', lambda: None)

    res = await ga.reconcile_on_startup(train_jobs_dir=jobs_dir, sentinel=sentinel)
    # No active jobs found -> reconcile falls through to the "nothing
    # active" branch, which unconditionally calls start_gpu_services() as
    # the backstop that brings configured containers back up. With no
    # containers configured that is a pure no-op regardless of docker
    # availability.
    assert res.action == 'noop'


@pytest.mark.asyncio
async def test_reconcile_skips_corrupt_status_file(tmp_path: Path, monkeypatch):
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    (jobs_dir / 'busted.status.json').write_text('not json')
    sentinel = tmp_path / 'pause.sentinel'

    monkeypatch.setattr(ga, '_docker_client', lambda: None)

    res = await ga.reconcile_on_startup(train_jobs_dir=jobs_dir, sentinel=sentinel)
    # Corrupt files are skipped, so no 'active' job is detected -> same
    # "nothing active" backstop path as above -> 'noop'.
    assert res.action == 'noop'

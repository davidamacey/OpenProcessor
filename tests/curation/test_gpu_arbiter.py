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

from src.config import GpuArbiterConfig, gpu_arbiter as gpu_arbiter_config_module
from src.services.training import gpu_arbiter as ga


if TYPE_CHECKING:
    from pathlib import Path


def _set_config(monkeypatch: pytest.MonkeyPatch, cfg: GpuArbiterConfig) -> None:
    monkeypatch.setattr(gpu_arbiter_config_module, '_default_gpu_arbiter_config', cfg)


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


def test_needs_multi_gpu_stop_alias_removed():
    """The private-deployment ``needs_gemma_stop`` alias is gone entirely --
    stop decisions are GPU-scope-driven now, not a "gemma" special case."""
    assert not hasattr(ga, 'needs_gemma_stop')


# NOTE: the reference test suite had a `test_needs_gemma_stop_single_gpu2_true`
# pinning a deployment-specific fact -- "GPU 2 always hosts the Gemma vLLM
# server, so a lone '2' claim must still stop it." That knowledge is now
# expressed generically: a container scoped to GPU 2 (``name@2`` in
# ``OP_GPU_ARBITER_CONTAINERS``) is stopped by any claim that intersects
# GPU 2, single- or multi-GPU. See the ``containers_to_stop`` tests below.


# ---------------------------------------------------------------------------
# containers_to_stop / needs_service_stop
# ---------------------------------------------------------------------------


def test_containers_to_stop_scoped_hit(monkeypatch):
    cfg = GpuArbiterConfig(container_gpus=(('vllm-server', frozenset({2})),))
    _set_config(monkeypatch, cfg)
    assert ga.containers_to_stop('2') == ('vllm-server',)
    assert ga.containers_to_stop('0,2') == ('vllm-server',)


def test_containers_to_stop_scoped_miss(monkeypatch):
    cfg = GpuArbiterConfig(container_gpus=(('vllm-server', frozenset({2})),))
    _set_config(monkeypatch, cfg)
    assert ga.containers_to_stop('0') == ()


def test_containers_to_stop_unscoped_single_gpu_untouched(monkeypatch):
    cfg = GpuArbiterConfig(container_gpus=(('region-worker', None),))
    _set_config(monkeypatch, cfg)
    assert ga.containers_to_stop('0') == ()


def test_containers_to_stop_unscoped_multi_gpu_stopped(monkeypatch):
    cfg = GpuArbiterConfig(container_gpus=(('region-worker', None),))
    _set_config(monkeypatch, cfg)
    assert ga.containers_to_stop('0,2') == ('region-worker',)


def test_containers_to_stop_mixed_scoped_and_unscoped(monkeypatch):
    cfg = GpuArbiterConfig(
        container_gpus=(('vllm-server', frozenset({2})), ('region-worker', None))
    )
    _set_config(monkeypatch, cfg)
    # Single-GPU claim on GPU 2: scoped container stops, unscoped doesn't.
    assert ga.containers_to_stop('2') == ('vllm-server',)
    # Multi-GPU claim spanning both: both stop, in configured order.
    assert ga.containers_to_stop('0,2') == ('vllm-server', 'region-worker')


def test_containers_to_stop_none_configured(monkeypatch):
    _set_config(monkeypatch, GpuArbiterConfig())
    assert ga.containers_to_stop('0,2') == ()


def test_needs_service_stop_matches_containers_to_stop(monkeypatch):
    cfg = GpuArbiterConfig(container_gpus=(('vllm-server', frozenset({2})),))
    _set_config(monkeypatch, cfg)
    assert ga.needs_service_stop('2') is True
    assert ga.needs_service_stop('0') is False


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
async def test_claim_dual_gpu_fails_closed_without_docker(tmp_path: Path, monkeypatch):
    """S-5: dual-GPU on a host without a usable docker SDK/socket must
    refuse the claim (GpuArbiterStopFailedError), not silently fall back
    to a sentinel-only pause -- a sentinel pauses a paired *worker*
    process, not a sibling container sharing the GPU (e.g. a large vLLM
    process), so falling back would let training start right next to it.
    Only reachable once at least one container is configured -- otherwise
    stop_gpu_services() short-circuits to a pure no-op before ever
    checking docker (see test_gpu_arbiter_config.py). The training lock
    this call wrote must also be cleared on the way out, so a refused
    claim never leaves stale lock state behind.
    """
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    monkeypatch.setattr(ga, '_docker_client', lambda: None)

    fake_cfg = GpuArbiterConfig(
        containers=('fake-gpu-service',),
        container_gpus=(('fake-gpu-service', None),),
    )
    _set_config(monkeypatch, fake_cfg)

    with pytest.raises(ga.GpuArbiterStopFailedError):
        await ga.claim_gpus_for_training('0,2')
    assert not ga.sentinel_path().exists()
    assert ga.read_training_lock() is None


class _FakeContainer:
    def __init__(self, name: str, status: str = 'running') -> None:
        self.name = name
        self.status = status

    def stop(self, timeout: int = 30) -> None:  # noqa: ARG002 - matches docker SDK signature
        self.status = 'exited'

    def start(self) -> None:
        self.status = 'running'

    def reload(self) -> None:
        return None


class _FakeContainers:
    def __init__(self, registry: dict[str, _FakeContainer]) -> None:
        self._registry = registry

    def get(self, name: str) -> _FakeContainer:
        import docker.errors

        try:
            return self._registry[name]
        except KeyError as exc:
            raise docker.errors.NotFound(name) from exc


class _FakeDockerClient:
    def __init__(self, registry: dict[str, _FakeContainer]) -> None:
        self.containers = _FakeContainers(registry)


@pytest.mark.asyncio
async def test_claim_single_gpu_scoped_container_stops_it(tmp_path: Path, monkeypatch):
    """A single-GPU claim that intersects a *scoped* container's GPU set
    stops that container even though it's not a multi-GPU claim -- this is
    the whole point of GPU-scoped containers (task #1 in the plan)."""
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    registry = {'vllm-server': _FakeContainer('vllm-server')}
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))

    fake_cfg = GpuArbiterConfig(
        containers=('vllm-server',),
        container_gpus=(('vllm-server', frozenset({2})),),
    )
    _set_config(monkeypatch, fake_cfg)

    res = await ga.claim_gpus_for_training('2')
    assert res.action == 'gpu_services_stopped'
    assert registry['vllm-server'].status == 'exited'
    # Sentinel is also set (belt-and-suspenders) so a paired worker that
    # comes back up mid-run still pauses.
    assert ga.sentinel_path().exists()


@pytest.mark.asyncio
async def test_release_single_gpu_scoped_container_restarts_it(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    registry = {'vllm-server': _FakeContainer('vllm-server', status='exited')}
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))

    fake_cfg = GpuArbiterConfig(
        containers=('vllm-server',),
        container_gpus=(('vllm-server', frozenset({2})),),
    )
    _set_config(monkeypatch, fake_cfg)

    res = await ga.release_gpus_after_training('2')
    assert res.action == 'gpu_services_started'
    assert registry['vllm-server'].status == 'running'
    assert not ga.sentinel_path().exists()


@pytest.mark.asyncio
async def test_claim_single_gpu_unscoped_container_untouched(tmp_path: Path, monkeypatch):
    """A single-GPU claim must not stop a container scoped to a *different*
    GPU, nor an unscoped container (unscoped only stops on multi-GPU)."""
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    registry = {
        'vllm-server': _FakeContainer('vllm-server'),
        'region-worker': _FakeContainer('region-worker'),
    }
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))

    fake_cfg = GpuArbiterConfig(
        containers=('vllm-server', 'region-worker'),
        container_gpus=(('vllm-server', frozenset({2})), ('region-worker', None)),
    )
    _set_config(monkeypatch, fake_cfg)

    res = await ga.claim_gpus_for_training('0')
    assert res.action == 'sentinel_set'
    assert registry['vllm-server'].status == 'running'
    assert registry['region-worker'].status == 'running'


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


# ---------------------------------------------------------------------------
# reconcile_on_startup -- GPU-scoped containers (union across active runs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_keeps_scoped_container_stopped_while_active(tmp_path: Path, monkeypatch):
    """A single-GPU run on the scoped container's GPU keeps it stopped, and
    starts every OTHER configured container that isn't in the claim's
    GPU-intersecting set."""
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    registry = {
        'vllm-server': _FakeContainer('vllm-server', status='exited'),
        'other-service': _FakeContainer('other-service', status='exited'),
    }
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))
    fake_cfg = GpuArbiterConfig(
        containers=('vllm-server', 'other-service'),
        container_gpus=(('vllm-server', frozenset({2})), ('other-service', None)),
    )
    _set_config(monkeypatch, fake_cfg)

    _write_job(jobs_dir, 'job-a', cuda_visible_devices='2')
    _write_status(jobs_dir, 'job-a', 'running')

    sentinel = tmp_path / 'pause.sentinel'
    res = await ga.reconcile_on_startup(train_jobs_dir=jobs_dir, sentinel=sentinel)
    assert res.action == 'gpu_services_started'
    # vllm-server stays down (scoped to the claimed GPU); other-service, not
    # in the claim's intersecting set, comes back up.
    assert registry['vllm-server'].status == 'exited'
    assert registry['other-service'].status == 'running'


@pytest.mark.asyncio
async def test_reconcile_starts_scoped_container_when_idle(tmp_path: Path, monkeypatch):
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    registry = {'vllm-server': _FakeContainer('vllm-server', status='exited')}
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))
    fake_cfg = GpuArbiterConfig(
        containers=('vllm-server',),
        container_gpus=(('vllm-server', frozenset({2})),),
    )
    _set_config(monkeypatch, fake_cfg)

    sentinel = tmp_path / 'pause.sentinel'
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()

    res = await ga.reconcile_on_startup(train_jobs_dir=jobs_dir, sentinel=sentinel)
    assert res.action == 'gpu_services_started'
    assert registry['vllm-server'].status == 'running'
    assert not sentinel.exists()

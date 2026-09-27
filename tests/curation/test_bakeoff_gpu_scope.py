"""Tests for OP_BAKEOFF_HOST_GPUS scoping (W5.1).

A bake-off enqueue historically stopped *every* configured GPU-resident
container (``gpu_arbiter.stop_gpu_services()`` with no ``containers=``
argument), even when the evaluator only claims one host GPU. That means a
bake-off scored on an idle GPU still stopped a service pinned to a
*different* GPU (e.g. vLLM on GPU 2) for no reason.

``GpuArbiterConfig.bakeoff_host_gpus`` (``OP_BAKEOFF_HOST_GPUS``) names the
host GPU ids the evaluator container is actually attached to. When set, the
router (and the reconcile loop, while a bake-off is queued) only stop
containers whose configured GPU scope (``containers_to_stop``) intersects
those ids. Unset keeps the old, conservative "stop everything configured"
behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config import GpuArbiterConfig, gpu_arbiter as gpu_arbiter_config_module
from src.services.training import gpu_arbiter as ga


if TYPE_CHECKING:
    from pathlib import Path


def _set_config(monkeypatch: pytest.MonkeyPatch, cfg: GpuArbiterConfig) -> None:
    monkeypatch.setattr(gpu_arbiter_config_module, '_default_gpu_arbiter_config', cfg)


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


def _enqueue_stop_containers(cfg: GpuArbiterConfig) -> tuple[str, ...] | None:
    """Mirrors the router's ``containers=`` argument at enqueue time."""
    scope = cfg.bakeoff_host_gpus
    return ga.containers_to_stop(scope) if scope else None


@pytest.mark.asyncio
async def test_bakeoff_host_gpu_disjoint_from_service_scope_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluator on host GPU 0; a service scoped to GPU 2 must not be
    touched, and the docker SDK must never even be constructed."""

    def _boom() -> None:
        raise AssertionError('docker client must not be constructed for a disjoint scope')

    monkeypatch.setattr(ga, '_docker_client', _boom)
    cfg = GpuArbiterConfig(
        containers=('svc',),
        container_gpus=(('svc', frozenset({2})),),
        bakeoff_host_gpus='0',
    )
    _set_config(monkeypatch, cfg)

    action = await ga.stop_gpu_services(containers=_enqueue_stop_containers(cfg))
    assert action.action == 'noop'


@pytest.mark.asyncio
async def test_bakeoff_host_gpu_intersects_service_scope_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluator on host GPU 2 -- the service pinned there is stopped."""
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    registry = {'svc': _FakeContainer('svc')}
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))
    cfg = GpuArbiterConfig(
        containers=('svc',),
        container_gpus=(('svc', frozenset({2})),),
        bakeoff_host_gpus='2',
    )
    _set_config(monkeypatch, cfg)

    action = await ga.stop_gpu_services(containers=_enqueue_stop_containers(cfg))
    assert action.action == 'gpu_services_stopped'
    assert registry['svc'].status == 'exited'


@pytest.mark.asyncio
async def test_bakeoff_host_gpu_unset_keeps_back_compat_stop_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OP_BAKEOFF_HOST_GPUS unset must keep today's behavior: stop every
    configured container regardless of scope."""
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    registry = {'svc': _FakeContainer('svc')}
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))
    cfg = GpuArbiterConfig(
        containers=('svc',),
        container_gpus=(('svc', frozenset({2})),),
        bakeoff_host_gpus=None,
    )
    _set_config(monkeypatch, cfg)

    action = await ga.stop_gpu_services(containers=_enqueue_stop_containers(cfg))
    assert action.action == 'gpu_services_stopped'
    assert registry['svc'].status == 'exited'


@pytest.mark.asyncio
async def test_reconcile_restarts_service_outside_bakeoff_host_gpu_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued bake-off scoped to host GPU 0 must not keep a GPU-2-scoped
    service down: reconcile only holds down the intersecting subset,
    releasing everything else (e.g. after a training run on GPU 2 that
    claimed it releases)."""
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    bakeoff_jobs_dir = tmp_path / 'bakeoff_jobs'
    bakeoff_jobs_dir.mkdir()
    (bakeoff_jobs_dir / 'b1.job.json').write_text('{}')

    registry = {'svc': _FakeContainer('svc', status='exited')}
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))
    cfg = GpuArbiterConfig(
        containers=('svc',),
        container_gpus=(('svc', frozenset({2})),),
        bakeoff_host_gpus='0',
        bakeoff_jobs_dir=str(bakeoff_jobs_dir),
    )
    _set_config(monkeypatch, cfg)

    sentinel = tmp_path / 'pause.sentinel'
    res = await ga.reconcile_on_startup(train_jobs_dir=jobs_dir, sentinel=sentinel)
    assert res.action == 'gpu_services_started'
    assert registry['svc'].status == 'running'


@pytest.mark.asyncio
async def test_reconcile_keeps_intersecting_service_down_during_bakeoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued bake-off scoped to host GPU 2 keeps a GPU-2-scoped service
    stopped."""
    monkeypatch.setattr(ga, '_state_dir', lambda: tmp_path)
    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    bakeoff_jobs_dir = tmp_path / 'bakeoff_jobs'
    bakeoff_jobs_dir.mkdir()
    (bakeoff_jobs_dir / 'b1.job.json').write_text('{}')

    registry = {'svc': _FakeContainer('svc', status='running')}
    monkeypatch.setattr(ga, '_docker_client', lambda: _FakeDockerClient(registry))
    cfg = GpuArbiterConfig(
        containers=('svc',),
        container_gpus=(('svc', frozenset({2})),),
        bakeoff_host_gpus='2',
        bakeoff_jobs_dir=str(bakeoff_jobs_dir),
    )
    _set_config(monkeypatch, cfg)

    sentinel = tmp_path / 'pause.sentinel'
    res = await ga.reconcile_on_startup(train_jobs_dir=jobs_dir, sentinel=sentinel)
    assert res.action == 'gpu_services_stopped'
    assert registry['svc'].status == 'exited'

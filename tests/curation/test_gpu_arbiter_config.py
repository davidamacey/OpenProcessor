"""Pins for ``GpuArbiterConfig`` (Chunk 6).

See ``docs/design/oss_genericization_phase2_plan.md`` Chunk 6 — a
generic install with no configured containers/GPU ids must degrade to a
no-op, not crash.
"""

from __future__ import annotations

import asyncio

from src.config import GpuArbiterConfig, get_gpu_arbiter_config
from src.services.training import gpu_arbiter


def test_defaults_are_empty_and_permissive() -> None:
    cfg = GpuArbiterConfig()
    assert cfg.allowed_gpu_ids == frozenset()
    assert cfg.containers == ()
    assert cfg.trainer_container is None
    assert cfg.bakeoff_jobs_dir is None


def test_is_gpu_allowed_permissive_when_unset() -> None:
    cfg = GpuArbiterConfig()
    assert cfg.is_gpu_allowed(0) is True
    assert cfg.is_gpu_allowed(7) is True


def test_is_gpu_allowed_restricts_when_configured() -> None:
    cfg = GpuArbiterConfig(allowed_gpu_ids=frozenset({0, 2}))
    assert cfg.is_gpu_allowed(0) is True
    assert cfg.is_gpu_allowed(1) is False


def test_get_gpu_arbiter_config_singleton() -> None:
    assert get_gpu_arbiter_config() is get_gpu_arbiter_config()


def test_stop_gpu_services_is_noop_with_no_configured_containers() -> None:
    """An install with GpuArbiterConfig(containers=()) must not touch the
    docker SDK at all -- it degrades to a no-op rather than crashing or
    warning about a missing socket."""
    action = asyncio.run(gpu_arbiter.stop_gpu_services(containers=()))
    assert action.action == 'noop'


def test_start_gpu_services_is_noop_with_no_configured_containers() -> None:
    action = asyncio.run(gpu_arbiter.start_gpu_services(containers=()))
    assert action.action == 'noop'


def test_probe_trainer_reachable_skips_when_unconfigured() -> None:
    reachable, detail = asyncio.run(gpu_arbiter.probe_trainer_reachable(container_name=None))
    assert reachable is True
    assert 'not configured' in detail or 'not applicable' in detail

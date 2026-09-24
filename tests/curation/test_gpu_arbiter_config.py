"""Pins for ``GpuArbiterConfig`` (Chunk 6).

See ``docs/design/curation_design_rationale.md`` for the
config-driven-genericity design principle this follows — a generic
install with no configured containers/GPU ids must degrade to a no-op,
not crash.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from src.config import GpuArbiterConfig, get_gpu_arbiter_config
from src.services.training import gpu_arbiter


if TYPE_CHECKING:
    from pathlib import Path


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


# =============================================================================
# GpuArbiterConfig.from_env (cutover plan: GPU policy as config, not code)
# =============================================================================

_ARBITER_ENV_VARS = (
    'OP_GPU_ALLOWED_IDS',
    'OP_GPU_ARBITER_CONTAINERS',
    'OP_GPU_ARBITER_TRAINER_CONTAINER',
    'OP_BAKEOFF_JOBS_DIR',
)


@pytest.fixture
def clean_arbiter_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    for name in _ARBITER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Force the singleton to be rebuilt from the (patched) env.
    monkeypatch.setattr(gpu_arbiter_config_module, '_default_gpu_arbiter_config', None)
    return monkeypatch


def test_from_env_unset_is_permissive_default(clean_arbiter_env: pytest.MonkeyPatch) -> None:
    assert GpuArbiterConfig.from_env() == GpuArbiterConfig()


def test_from_env_parses_every_field(clean_arbiter_env: pytest.MonkeyPatch) -> None:
    clean_arbiter_env.setenv('OP_GPU_ALLOWED_IDS', ' 2, 0 ,')
    clean_arbiter_env.setenv('OP_GPU_ARBITER_CONTAINERS', 'vlm-server, region-worker')
    clean_arbiter_env.setenv('OP_GPU_ARBITER_TRAINER_CONTAINER', 'trainer')
    clean_arbiter_env.setenv('OP_BAKEOFF_JOBS_DIR', '/var/lib/openprocessor/bakeoff_jobs')
    cfg = GpuArbiterConfig.from_env()
    assert cfg.allowed_gpu_ids == frozenset({0, 2})
    assert cfg.containers == ('vlm-server', 'region-worker')
    assert cfg.trainer_container == 'trainer'
    assert cfg.bakeoff_jobs_dir == '/var/lib/openprocessor/bakeoff_jobs'


def test_from_env_empty_strings_mean_unset(clean_arbiter_env: pytest.MonkeyPatch) -> None:
    for name in _ARBITER_ENV_VARS:
        clean_arbiter_env.setenv(name, '')
    assert GpuArbiterConfig.from_env() == GpuArbiterConfig()


@pytest.mark.parametrize('raw', ['0,one', '0;2', '-1'])
def test_from_env_malformed_allowlist_raises(
    clean_arbiter_env: pytest.MonkeyPatch, raw: str
) -> None:
    """A typo in the GPU fence must not silently become 'unrestricted'."""
    clean_arbiter_env.setenv('OP_GPU_ALLOWED_IDS', raw)
    with pytest.raises(ValueError, match='OP_GPU_ALLOWED_IDS'):
        GpuArbiterConfig.from_env()


def test_default_singleton_is_built_from_env(clean_arbiter_env: pytest.MonkeyPatch) -> None:
    clean_arbiter_env.setenv('OP_GPU_ALLOWED_IDS', '0,2')
    clean_arbiter_env.setenv('OP_GPU_ARBITER_TRAINER_CONTAINER', 'trainer')
    cfg = get_gpu_arbiter_config()
    assert cfg.allowed_gpu_ids == frozenset({0, 2})
    assert cfg.trainer_container == 'trainer'


def test_env_allowlist_rejects_disallowed_gpu_on_training_start(
    clean_arbiter_env: pytest.MonkeyPatch,
) -> None:
    """End to end through the real consumer: a training job spec (what
    POST /train/start validates) targeting a GPU outside
    OP_GPU_ALLOWED_IDS is rejected; one inside it is accepted."""
    from src.services.training.jobs import TrainJobSpec

    clean_arbiter_env.setenv('OP_GPU_ALLOWED_IDS', '0,2')
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='1')
    ok = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='2,0')
    assert ok.cuda_visible_devices == '0,2'


def test_env_bakeoff_jobs_dir_reaches_reconcile_check(
    clean_arbiter_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OP_BAKEOFF_JOBS_DIR must drive gpu_arbiter.bakeoff_active() -- the
    reconcile loop's 'is a bake-off queued' check -- with no code config."""
    clean_arbiter_env.setenv('OP_BAKEOFF_JOBS_DIR', str(tmp_path))
    assert gpu_arbiter.bakeoff_active() is False
    (tmp_path / 'x.job.json').write_text('{}')
    assert gpu_arbiter.bakeoff_active() is True

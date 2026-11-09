"""Pins for ``GpuArbiterConfig``.

See ``docs/design/curation_design_rationale.md`` for the
config-driven-genericity design principle this follows — a generic
install with no configured containers/GPU ids must degrade to a no-op,
not crash.
"""

from __future__ import annotations

import asyncio

import pytest

from src.config import GpuArbiterConfig, get_gpu_arbiter_config
from src.services.training import gpu_arbiter_docker


def test_defaults_are_empty_and_permissive() -> None:
    cfg = GpuArbiterConfig()
    assert cfg.allowed_gpu_ids == frozenset()
    assert cfg.containers == ()
    assert cfg.container_gpus == ()
    assert cfg.gpu_labels == {}
    assert cfg.default_train_gpus is None


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
    action = asyncio.run(gpu_arbiter_docker.stop_gpu_services(containers=()))
    assert action.action == 'noop'


def test_start_gpu_services_is_noop_with_no_configured_containers() -> None:
    action = asyncio.run(gpu_arbiter_docker.start_gpu_services(containers=()))
    assert action.action == 'noop'


def test_docker_client_unavailable_warns_once_not_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G-15: without docker-compose.gpu-arbiter.yml's socket mount, every
    claim/release/reconcile probe calls ``_docker_client()`` and gets
    ``None`` (fail-open). That must log exactly one warning per outage,
    not one per call -- otherwise a busy reconcile loop drowns the log."""

    class _Boom:
        def from_env(self) -> None:
            raise RuntimeError('no such file or directory: /var/run/docker.sock')

    monkeypatch.setattr(gpu_arbiter_docker, '_docker_unavailable_logged', [False])
    monkeypatch.setattr('docker.from_env', _Boom().from_env)

    warnings: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        gpu_arbiter_docker.logger,
        'warning',
        lambda event, **kw: warnings.append((event, kw)),
    )

    for _ in range(3):
        assert gpu_arbiter_docker._docker_client() is None

    assert len(warnings) == 1
    assert warnings[0][0] == 'arbiter_docker_unavailable'

    # Recovery clears the dedup flag so a later outage warns again.
    class _Ok:
        def ping(self) -> None:
            return None

    monkeypatch.setattr('docker.from_env', _Ok)
    assert gpu_arbiter_docker._docker_client() is not None

    monkeypatch.setattr('docker.from_env', _Boom().from_env)
    assert gpu_arbiter_docker._docker_client() is None
    assert len(warnings) == 2


# =============================================================================
# GpuArbiterConfig.from_env (GPU policy as config, not code)
# =============================================================================

_ARBITER_ENV_VARS = (
    'OP_GPU_ALLOWED_IDS',
    'OP_GPU_ARBITER_CONTAINERS',
    'OP_GPU_LABELS',
    'OP_TRAIN_DEFAULT_GPUS',
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
    clean_arbiter_env.setenv('OP_GPU_ARBITER_CONTAINERS', 'vlm-server@2, region-worker')
    clean_arbiter_env.setenv('OP_GPU_LABELS', '0=RTX A6000,2=RTX A6000')
    clean_arbiter_env.setenv('OP_TRAIN_DEFAULT_GPUS', '2')
    cfg = GpuArbiterConfig.from_env()
    assert cfg.allowed_gpu_ids == frozenset({0, 2})
    assert cfg.containers == ('vlm-server', 'region-worker')
    assert cfg.container_gpus == (('vlm-server', frozenset({2})), ('region-worker', None))
    assert cfg.gpu_labels == {0: 'RTX A6000', 2: 'RTX A6000'}
    assert cfg.default_train_gpus == '2'


def test_from_env_scoped_container_multi_gpu(clean_arbiter_env: pytest.MonkeyPatch) -> None:
    clean_arbiter_env.setenv('OP_GPU_ARBITER_CONTAINERS', 'segmenter@0/2')
    cfg = GpuArbiterConfig.from_env()
    assert cfg.container_gpus == (('segmenter', frozenset({0, 2})),)


@pytest.mark.parametrize(
    'raw',
    ['name@', 'name@x', 'name@-1', 'name@0/-1', '@2', 'name@0/x'],
)
def test_from_env_malformed_container_scope_raises(
    clean_arbiter_env: pytest.MonkeyPatch, raw: str
) -> None:
    clean_arbiter_env.setenv('OP_GPU_ARBITER_CONTAINERS', raw)
    with pytest.raises(ValueError, match='OP_GPU_ARBITER_CONTAINERS'):
        GpuArbiterConfig.from_env()


@pytest.mark.parametrize('raw', ['badentry', '0=', '-1=A6000', 'x=A6000'])
def test_from_env_malformed_gpu_labels_raises(
    clean_arbiter_env: pytest.MonkeyPatch, raw: str
) -> None:
    clean_arbiter_env.setenv('OP_GPU_LABELS', raw)
    with pytest.raises(ValueError, match='OP_GPU_LABELS'):
        GpuArbiterConfig.from_env()


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
    cfg = get_gpu_arbiter_config()
    assert cfg.allowed_gpu_ids == frozenset({0, 2})


def test_env_allowlist_rejects_disallowed_gpu_on_training_start(
    clean_arbiter_env: pytest.MonkeyPatch,
) -> None:
    """End to end through the real consumer: a training job spec (what
    POST /train/start validates) targeting a GPU outside
    OP_GPU_ALLOWED_IDS is rejected; one inside it is accepted."""
    from src.services.training.job_models import TrainJobSpec

    clean_arbiter_env.setenv('OP_GPU_ALLOWED_IDS', '0,2')
    with pytest.raises(ValueError, match='allowed GPU id'):
        TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='1')
    ok = TrainJobSpec(dataset_export_dir='/data/exports/x', cuda_visible_devices='2,0')
    assert ok.cuda_visible_devices == '0,2'

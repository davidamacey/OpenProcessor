"""Which configured GPU-resident services a claim stops (pure policy of the arbiter tick)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config import GpuArbiterConfig, gpu_arbiter as gpu_arbiter_config_module
from src.domain import gpu_claims


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch) -> Callable[[GpuArbiterConfig], None]:
    def _set(cfg: GpuArbiterConfig) -> None:
        monkeypatch.setattr(gpu_arbiter_config_module, '_default_gpu_arbiter_config', cfg)

    return _set


def test_parse_none_and_empty_are_no_devices() -> None:
    assert gpu_claims.parse_cuda_visible_devices(None) == []
    assert gpu_claims.parse_cuda_visible_devices('') == []


def test_parse_keeps_order_and_strips_whitespace() -> None:
    assert gpu_claims.parse_cuda_visible_devices('0') == [0]
    assert gpu_claims.parse_cuda_visible_devices('  2 ,  0 ') == [2, 0]


def test_parse_skips_an_unparseable_token() -> None:
    assert gpu_claims.parse_cuda_visible_devices('0,a,2') == [0, 2]


def test_a_scoped_service_is_stopped_by_any_claim_that_touches_its_gpu(configure) -> None:
    configure(GpuArbiterConfig(container_gpus=(('vllm-server', frozenset({2})),)))
    assert gpu_claims.containers_to_stop('2') == ('vllm-server',)
    assert gpu_claims.containers_to_stop('0,2') == ('vllm-server',)
    assert gpu_claims.containers_to_stop('0') == ()


def test_an_unscoped_service_is_stopped_only_by_a_multi_gpu_claim(configure) -> None:
    configure(GpuArbiterConfig(container_gpus=(('region-worker', None),)))
    assert gpu_claims.containers_to_stop('0') == ()
    assert gpu_claims.containers_to_stop('0,2') == ('region-worker',)


def test_scoped_and_unscoped_stop_in_the_configured_order(configure) -> None:
    configure(
        GpuArbiterConfig(container_gpus=(('vllm-server', frozenset({2})), ('region-worker', None)))
    )
    assert gpu_claims.containers_to_stop('2') == ('vllm-server',)
    assert gpu_claims.containers_to_stop('0,2') == ('vllm-server', 'region-worker')


def test_nothing_configured_stops_nothing(configure) -> None:
    configure(GpuArbiterConfig())
    assert gpu_claims.containers_to_stop('0,2') == ()
    assert gpu_claims.needs_service_stop('0,2') is False


def test_needs_service_stop_matches_containers_to_stop(configure) -> None:
    configure(GpuArbiterConfig(container_gpus=(('vllm-server', frozenset({2})),)))
    assert gpu_claims.needs_service_stop('2') is True
    assert gpu_claims.needs_service_stop('0') is False


def test_a_train_or_probe_claim_goes_by_its_own_devices(configure) -> None:
    configure(
        GpuArbiterConfig(
            containers=('vllm-server', 'region-worker'),
            container_gpus=(('vllm-server', frozenset({2})), ('region-worker', None)),
        )
    )
    assert gpu_claims.claim_services('train', '2') == ('vllm-server',)
    assert gpu_claims.claim_services('probe', '0') == ()


def test_a_claim_with_unknown_devices_claims_every_configured_service(configure) -> None:
    configure(
        GpuArbiterConfig(
            containers=('vllm-server', 'region-worker'),
            container_gpus=(('vllm-server', frozenset({2})), ('region-worker', None)),
        )
    )
    assert gpu_claims.claim_services('train', None) == ('vllm-server', 'region-worker')


def test_a_bakeoff_claims_every_service_without_a_host_scope(configure) -> None:
    configure(
        GpuArbiterConfig(
            containers=('vllm-server', 'region-worker'),
            container_gpus=(('vllm-server', frozenset({2})), ('region-worker', None)),
        )
    )
    assert gpu_claims.claim_services('bakeoff', None) == ('vllm-server', 'region-worker')


def test_a_bakeoff_with_a_host_scope_stops_only_the_intersecting_services(configure) -> None:
    configure(
        GpuArbiterConfig(
            containers=('vllm-server', 'region-worker'),
            container_gpus=(('vllm-server', frozenset({2})), ('region-worker', None)),
            bakeoff_host_gpus='0',
        )
    )
    assert gpu_claims.claim_services('bakeoff', None) == ()
    configure(
        GpuArbiterConfig(
            containers=('vllm-server', 'region-worker'),
            container_gpus=(('vllm-server', frozenset({2})), ('region-worker', None)),
            bakeoff_host_gpus='2',
        )
    )
    assert gpu_claims.claim_services('bakeoff', None) == ('vllm-server',)

"""GPU device resolution and NVML telemetry host ids."""

from __future__ import annotations

import sys

import gpu_select
import job_status
import pytest


def test_gpu_ids_pass_through_when_no_order_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_TRAIN_GPU_ORDER', raising=False)
    assert gpu_select.resolve_local_device('2') == '2'
    assert gpu_select.resolve_local_device('0,2') == '0,2'
    assert gpu_select.resolve_local_device(None) is None
    assert gpu_select.resolve_local_device('') is None


def test_host_gpu_ids_map_to_positional_local_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container pinned to ['0','2'] sees host GPU 2 as local index 1.

    Deriving the device string from the *count* of requested GPUs would run a
    host-GPU-2-only request on host GPU 0 instead.
    """
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '0,2')
    assert gpu_select.resolve_local_device('0') == '0'
    assert gpu_select.resolve_local_device('2') == '1'
    assert gpu_select.resolve_local_device('0,2') == '0,1'


def test_unattached_gpu_request_fails_instead_of_mis_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '0,2')
    with pytest.raises(ValueError, match='not attached to this container'):
        gpu_select.resolve_local_device('1')


# =============================================================================
# NVML telemetry reports host ids
# =============================================================================


def test_gpu_telemetry_reports_host_index_through_gpu_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    class _H:
        def __init__(self, i: int) -> None:
            self.i = i

    nvml = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=_H,
        nvmlDeviceGetUtilizationRates=lambda _h: types.SimpleNamespace(gpu=50),
        nvmlDeviceGetMemoryInfo=lambda _h: types.SimpleNamespace(
            used=2 * 1024 * 1024, total=8 * 1024 * 1024
        ),
    )
    monkeypatch.setitem(sys.modules, 'pynvml', nvml)
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '2')
    assert [g['index'] for g in job_status._gpu_telemetry()] == [2]
    monkeypatch.delenv('OP_TRAIN_GPU_ORDER')
    assert [g['index'] for g in job_status._gpu_telemetry()] == [0]

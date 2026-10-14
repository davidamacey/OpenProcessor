"""GPU device resolution, the trainer capabilities file and NVML telemetry host ids."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import gpu_select
import job_status
import pytest


if TYPE_CHECKING:
    from pathlib import Path


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
# Trainer capabilities file (the API's preflight reads this)
# =============================================================================


def test_write_trainer_capabilities_publishes_env_gpu_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_TRAIN_GPU_ORDER', '0,2')
    monkeypatch.setenv('OP_BUILD_SHA', 'abc1234')
    jobs_dir = tmp_path / 'jobs'

    gpu_select.write_trainer_capabilities(jobs_dir)

    payload = json.loads((jobs_dir / gpu_select.TRAINER_CAPABILITIES_FILENAME).read_text())
    assert payload['gpu_order'] == [0, 2]
    assert payload['build_sha'] == 'abc1234'
    assert 'written_at' in payload
    assert 'visible_count' in payload


def test_write_trainer_capabilities_empty_order_is_unrestricted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_TRAIN_GPU_ORDER', raising=False)
    jobs_dir = tmp_path / 'jobs'

    gpu_select.write_trainer_capabilities(jobs_dir)

    payload = json.loads((jobs_dir / gpu_select.TRAINER_CAPABILITIES_FILENAME).read_text())
    assert payload['gpu_order'] == []


def test_write_trainer_capabilities_survives_cuda_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing/broken torch install (in the probe subprocess) must not
    crash trainer startup -- the file still gets written with
    visible_count=None."""
    import subprocess

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(1, args[0] if args else 'probe')

    monkeypatch.setattr(subprocess, 'run', _boom)
    jobs_dir = tmp_path / 'jobs'

    gpu_select.write_trainer_capabilities(jobs_dir)

    payload = json.loads((jobs_dir / gpu_select.TRAINER_CAPABILITIES_FILENAME).read_text())
    assert payload['visible_count'] is None


class TestCudaProbeIsolation:
    """The idle daemon process must never itself call into CUDA."""

    def test_probe_runs_in_a_subprocess_not_in_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Patching this process's own `import torch` must NOT affect the
        probe's result -- it must go through subprocess.run, which spawns
        a fresh Python interpreter unaffected by this process's sys.modules
        or builtins."""
        import builtins

        real_import = builtins.__import__

        def _boom(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == 'torch':
                msg = 'this process must never import torch for the probe'
                raise AssertionError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _boom)

        # The real subprocess (a fresh interpreter) is unaffected by the
        # patch above; if this venv has torch installed the probe returns
        # a real int, otherwise None -- either way, no AssertionError.
        result = gpu_select._probe_cuda_device_count()
        assert result is None or isinstance(result, int)

    def test_probe_uses_a_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hung probe subprocess must not hang trainer startup forever."""
        import subprocess

        captured: dict[str, Any] = {}

        def _fake_run(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            raise subprocess.TimeoutExpired(cmd='probe', timeout=kwargs.get('timeout', 0))

        monkeypatch.setattr(subprocess, 'run', _fake_run)

        result = gpu_select._probe_cuda_device_count()

        assert result is None
        assert captured.get('timeout') is not None


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

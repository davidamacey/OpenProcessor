"""The detector export traces on the CPU in FP32, whatever the host's torch can see.

The toolchain used to run in a venv with CPU-only torch, so the ONNX trace was always a CPU,
FP32 one (``dynamic=True`` only traces on the CPU) and the FP16 engine came from ModelOpt later.
The export now runs in the main environment, whose CUDA torch reports a GPU: the settings must
not follow it, or the trace would become a GPU FP16 one.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

torch = pytest.importorskip('torch')


def test_the_onnx_trace_is_cpu_fp32_even_when_torch_sees_a_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yolo_export_settings

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    try:
        settings = importlib.reload(yolo_export_settings)
        assert settings.DEVICE == 'cpu'
        assert not hasattr(settings, 'HALF')
    finally:
        monkeypatch.undo()
        importlib.reload(yolo_export_settings)


def test_the_input_size_is_unchanged() -> None:
    import yolo_export_settings as settings

    assert settings.IMG_SIZE == 640

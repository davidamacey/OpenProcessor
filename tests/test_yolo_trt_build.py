"""The ONNX-export step runs Ultralytics with device='cpu' (yolo_export_settings.DEVICE), which sets
CUDA_VISIBLE_DEVICES=-1 as a side effect -- in the same process, this then blinds the TensorRT
builder created right after it ("CUDA initialization failure with error: 100"), even though the
engine build doesn't use torch's CUDA at all.

``setup_trt_builder`` still constructs a real ``trt.Builder`` (a real CUDA context) after the
env-var repair, which needs an actual GPU -- not something a unit test should require.
``yolo_trt_build.trt`` is patched with a bare stand-in so only the repair logic itself is under test
here. Skips cleanly wherever TensorRT is not installed (the export image has it).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


pytest.importorskip('tensorrt')

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import yolo_trt_build  # noqa: E402  # follows the importorskip and the sys.path setup


@pytest.fixture(autouse=True)
def _fake_trt(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_trt = MagicMock()
    fake_trt.Logger.INFO = 0
    monkeypatch.setattr(yolo_trt_build, 'trt', fake_trt)
    return fake_trt


def test_poisoned_cuda_visible_devices_is_cleared_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    yolo_trt_build.setup_trt_builder()
    assert 'CUDA_VISIBLE_DEVICES' not in yolo_trt_build.os.environ


def test_an_unset_cuda_visible_devices_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    yolo_trt_build.setup_trt_builder()
    assert 'CUDA_VISIBLE_DEVICES' not in yolo_trt_build.os.environ


def test_a_legitimate_gpu_pin_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the exact '-1' (Ultralytics' CPU-mode sentinel) is cleared -- an operator's own GPU
    selection must survive."""
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    yolo_trt_build.setup_trt_builder()
    assert yolo_trt_build.os.environ['CUDA_VISIBLE_DEVICES'] == '0'

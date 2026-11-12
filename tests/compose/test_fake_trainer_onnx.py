"""The fake trainer's ONNX passes every check a promote makes on a trained detector."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.services.training.onnx_head import nms_free_head_problem
from src.utils.uint8_input import add_uint8_input


PATH = Path(__file__).resolve().parents[2] / 'docker' / 'test' / 'fake_onnx.py'


def _build(tmp_path: Path) -> Path:
    spec = importlib.util.spec_from_file_location('fake_onnx_harness', PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_detector_onnx(tmp_path / 'weights' / 'best.onnx')


def test_the_onnx_has_the_nms_free_head_and_converts_to_a_uint8_serving_copy(
    tmp_path: Path,
) -> None:
    path = _build(tmp_path)
    assert nms_free_head_problem(path) is None
    served = add_uint8_input(path, tmp_path / 'model.onnx')
    session = ort.InferenceSession(str(served), providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].type == 'tensor(uint8)'
    (out,) = session.run(None, {'images': np.zeros((2, 3, 640, 640), dtype=np.uint8)})
    assert out.shape == (2, 300, 6)

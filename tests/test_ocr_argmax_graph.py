"""``export/ocr_graph.py``: the recogniser's arg-max inside the graph.

The engine ships ``[B, T]`` indices and scores instead of ``[B, T, 18710]`` probabilities. The
tail must give exactly what ``preds.argmax(axis=2)`` and ``preds.max(axis=2)`` give on the host
(the decode ``CTCLabelDecode`` used to do), and must leave no other graph output behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import ocr_graph


def _toy(path: Path, classes: int = 7) -> None:
    """``x`` (B,3,4,W) float -> mean over channels and height -> softmax over W copies -> ``[B, T, C]``."""
    nodes = [
        helper.make_node('ReduceMean', ['x'], ['m'], axes=[1, 2], keepdims=0),  # [B, W]
        helper.make_node('Unsqueeze', ['m', 'ax'], ['m3']),  # [B, W, 1]
        helper.make_node('Mul', ['m3', 'k'], ['logits']),  # [B, W, C]
        helper.make_node('Softmax', ['logits'], ['fetch_name_0'], axis=2),
    ]
    k = onnx.numpy_helper.from_array(np.arange(1, classes + 1, dtype=np.float32) - 3.0, 'k')
    ax = onnx.numpy_helper.from_array(np.array([2], np.int64), 'ax')
    graph = helper.make_graph(
        nodes,
        'toy',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['b', 3, 4, 'w'])],
        [helper.make_tensor_value_info('fetch_name_0', TensorProto.FLOAT, ['b', 'w', classes])],
        initializer=[k, ax],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 9
    onnx.save(model, str(path))


def test_argmax_outputs_equal_the_host_argmax_and_max(tmp_path: Path) -> None:
    src, dst = tmp_path / 'a.onnx', tmp_path / 'b.onnx'
    _toy(src)
    ocr_graph.add_argmax_outputs(src, dst)
    x = np.random.default_rng(0).normal(size=(2, 3, 4, 9)).astype(np.float32)
    full = ort.InferenceSession(str(src)).run(None, {'x': x})[0]
    idx, score = ort.InferenceSession(str(dst)).run(['indices', 'scores'], {'x': x})
    assert idx.dtype == np.int32
    assert idx.shape == full.shape[:2]
    assert np.array_equal(idx, full.argmax(axis=2))
    assert np.array_equal(score, full.max(axis=2))


def test_the_probability_output_is_gone(tmp_path: Path) -> None:
    src, dst = tmp_path / 'a.onnx', tmp_path / 'b.onnx'
    _toy(src)
    ocr_graph.add_argmax_outputs(src, dst)
    names = [o.name for o in onnx.load(str(dst)).graph.output]
    assert names == ['indices', 'scores']


def test_a_graph_with_two_outputs_is_refused(tmp_path: Path) -> None:
    src, dst = tmp_path / 'a.onnx', tmp_path / 'b.onnx'
    _toy(src)
    model = onnx.load(str(src))
    model.graph.output.append(helper.make_tensor_value_info('extra', TensorProto.FLOAT, None))
    onnx.save(model, str(src))
    with pytest.raises(ValueError, match='one output'):
        ocr_graph.add_argmax_outputs(src, dst)

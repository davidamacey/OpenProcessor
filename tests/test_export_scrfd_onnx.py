"""``export/scrfd_onnx.py`` must keep the batch dimension first in every SCRFD output.

The raw InsightFace graph ends each head in ``Transpose(perm=[2,3,0,1])`` then
``Reshape([-1, D])``, which folds the batch into the anchor axis. The patch fixes both; the
Transpose lookup once matched protobuf wrappers by ``id()``, which never matches under the
upb protobuf backend, so no Transpose was fixed and the engine returned garbage boxes
(zero faces on any photo). Runs on a tiny stand-in graph: no GPU, no TensorRT.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

scrfd_onnx = importlib.import_module('scrfd_onnx')


CELLS = 4  # H = W of the stand-in feature map
CHANNELS = 20  # divisible by every head width (1, 4, 10)


def _raw_scrfd_graph() -> Any:
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits, outputs = [], [], []
    names = iter(str(448 + 23 * i) for i in range(9))
    # Per stride the raw graph has a distinct anchor count in dim 0 (used to sort the outputs).
    for anchors in (32, 8, 2):
        for width in (1, 4, 10):
            out = next(names)
            nodes.append(helper.make_node('Transpose', ['x'], [f't{out}'], perm=[2, 3, 0, 1]))
            # raw_data, as the real checkpoint stores it
            inits.append(
                numpy_helper.from_array(np.array([-1, width], dtype=np.int64), f'shape{out}')
            )
            nodes.append(helper.make_node('Reshape', [f't{out}', f'shape{out}'], [out]))
            outputs.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, [anchors, width]))
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, ['b', CHANNELS, CELLS, CELLS])
    graph = helper.make_graph(nodes, 'scrfd_like', [x], outputs, initializer=inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def test_transposes_keep_batch_first() -> None:
    model = _raw_scrfd_graph()
    rename_map = scrfd_onnx._rename_outputs_semantically(model)
    transposes, reshapes = scrfd_onnx._find_output_transposes(model, rename_map)
    # Identity-free: apply the findings to a freshly parsed copy, whose protobuf wrappers differ.
    model = onnx.load_from_string(model.SerializeToString())
    scrfd_onnx._fix_transpose_perms(model, transposes)
    scrfd_onnx._fix_reshape_shapes(model, reshapes)
    scrfd_onnx._set_batched_outputs(model, rename_map)

    session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    batch = np.random.default_rng(0).standard_normal((2, CHANNELS, CELLS, CELLS)).astype(np.float32)
    for out, meta in zip(session.run(None, {'x': batch}), session.get_outputs(), strict=True):
        width = out.shape[-1]
        assert out.shape == (2, CELLS * CELLS * CHANNELS // width, width), meta.name

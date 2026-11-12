"""A tiny valid detector ONNX for the fake trainer, so a promote reaches its success path.

One graph, built with ``onnx.helper`` (https://onnx.ai/onnx/api/helper.html): a float32 NCHW
``images`` input with a dynamic batch, a mean over the pixels, and a multiply by a constant
``[1, 300, 6]`` tensor, giving the single ``[N, 300, 6]`` ``output0`` the NMS-free YOLO26 head has
(``src/services/training/onnx_head.py`` checks exactly that, and ``src/utils/uint8_input.py`` needs
the one float32 input). Opset 13 keeps ``ReduceMean`` on its ``axes`` attribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


if TYPE_CHECKING:
    from pathlib import Path

TOPK = 300
DET_DIMS = 6
SIZE = 640


def build_detector_onnx(path: Path) -> Path:
    """Write the graph to ``path`` and return it."""
    images = helper.make_tensor_value_info('images', TensorProto.FLOAT, ['batch', 3, SIZE, SIZE])
    output = helper.make_tensor_value_info('output0', TensorProto.FLOAT, ['batch', TOPK, DET_DIMS])
    template = np.zeros((1, TOPK, DET_DIMS), dtype=np.float32)
    template[0, 0] = [0.2 * SIZE, 0.2 * SIZE, 0.8 * SIZE, 0.8 * SIZE, 0.9, 0.0]
    nodes = [
        helper.make_node('ReduceMean', ['images'], ['mean'], axes=[1, 2, 3], keepdims=0),
        helper.make_node('Reshape', ['mean', 'shape'], ['scale']),
        helper.make_node('Mul', ['scale', 'template'], ['output0']),
    ]
    graph = helper.make_graph(
        nodes,
        'fake_detector',
        [images],
        [output],
        initializer=[
            numpy_helper.from_array(np.array([-1, 1, 1], dtype=np.int64), 'shape'),
            numpy_helper.from_array(template, 'template'),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path

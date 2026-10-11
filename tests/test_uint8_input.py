"""``export/uint8_input.py``: the in-graph prefix that turns a float image input into uint8.

The graph keeps its input name, rank and dynamic batch; a uint8 pixel run through the
prefix must equal the value the CPU lookup table used to build, so the model's outputs
do not move.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import uint8_input  # noqa: E402


def _model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    return model


def _toy_model(path: Path, *, size: int = 4) -> None:
    """``images`` (B,3,S,S) float -> Mul by a constant -> ``out``; two consumers of the input."""
    scale = numpy_helper.from_array(np.full((1, 3, 1, 1), 2.0, dtype=np.float32), 'k')
    nodes = [
        helper.make_node('Mul', ['images', 'k'], ['scaled']),
        helper.make_node('Add', ['scaled', 'images'], ['out']),
    ]
    graph = helper.make_graph(
        nodes,
        'toy',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, ['batch', 3, size, size])],
        [helper.make_tensor_value_info('out', TensorProto.FLOAT, ['batch', 3, size, size])],
        initializer=[scale],
    )
    onnx.save(_model(graph), str(path))


def _run(path: Path, array: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    (out,) = session.run(None, {'images': array})
    return out


def _lookup_table(mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    """The CPU table the FP32 path used: ``(v / 255 - mean) / std`` per channel."""
    values = np.arange(256, dtype=np.float32) / np.float32(255.0)
    mean_a = np.asarray(mean, dtype=np.float32)
    std_a = np.asarray(std, dtype=np.float32)
    return (values[:, None] - mean_a) / std_a


class TestAddUint8Input:
    def test_input_becomes_uint8_with_name_and_dynamic_batch_kept(self, tmp_path: Path) -> None:
        src, dst = tmp_path / 'f.onnx', tmp_path / 'u.onnx'
        _toy_model(src)
        uint8_input.add_uint8_input(src, dst)
        graph = onnx.load(str(dst)).graph
        (inp,) = graph.input
        assert inp.name == 'images'
        assert inp.type.tensor_type.elem_type == TensorProto.UINT8
        dims = inp.type.tensor_type.shape.dim
        assert dims[0].dim_param == 'batch'
        assert [d.dim_value for d in dims[1:]] == [3, 4, 4]
        onnx.checker.check_model(onnx.load(str(dst)))

    def test_unit_range_matches_the_float_model_on_divided_pixels(self, tmp_path: Path) -> None:
        src, dst = tmp_path / 'f.onnx', tmp_path / 'u.onnx'
        _toy_model(src)
        uint8_input.add_uint8_input(src, dst)
        pixels = np.random.default_rng(0).integers(0, 256, size=(2, 3, 4, 4), dtype=np.uint8)
        expected = _run(src, pixels.astype(np.float32) / np.float32(255.0))
        got = _run(dst, pixels)
        np.testing.assert_array_equal(got, expected)  # the same bits, not just close

    def test_mean_std_matches_the_lookup_table_for_every_pixel_value(self, tmp_path: Path) -> None:
        src, dst = tmp_path / 'f.onnx', tmp_path / 'u.onnx'
        _toy_model(src, size=16)
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        uint8_input.add_uint8_input(src, dst, mean=mean, std=std)
        pixels = np.arange(256, dtype=np.uint8).reshape(1, 1, 16, 16).repeat(3, axis=1)
        table = _lookup_table(mean, std)
        normalized = np.stack([table[pixels[0, c], c] for c in range(3)])[None]
        expected = _run(src, np.ascontiguousarray(normalized, dtype=np.float32))
        np.testing.assert_array_equal(_run(dst, pixels), expected)  # the same bits, not just close

    def test_refuses_a_graph_whose_input_is_not_float(self, tmp_path: Path) -> None:
        src = tmp_path / 'f.onnx'
        graph = helper.make_graph(
            [helper.make_node('Identity', ['images'], ['out'])],
            'ints',
            [helper.make_tensor_value_info('images', TensorProto.INT64, [1])],
            [helper.make_tensor_value_info('out', TensorProto.INT64, [1])],
        )
        onnx.save(_model(graph), str(src))
        with pytest.raises(ValueError, match='float'):
            uint8_input.add_uint8_input(src, tmp_path / 'u.onnx')

    def test_refuses_to_run_twice(self, tmp_path: Path) -> None:
        src, once, twice = tmp_path / 'f.onnx', tmp_path / 'u.onnx', tmp_path / 'uu.onnx'
        _toy_model(src)
        uint8_input.add_uint8_input(src, once)
        with pytest.raises(ValueError, match='float'):
            uint8_input.add_uint8_input(once, twice)

    def test_fp16_baked_graph_with_a_cast_after_the_input_still_works(self, tmp_path: Path) -> None:
        """``keep_io_types`` leaves ``input -> Cast(fp16)``: the prefix goes in front of it."""
        src, dst = tmp_path / 'f.onnx', tmp_path / 'u.onnx'
        graph = helper.make_graph(
            [
                helper.make_node('Cast', ['images'], ['half'], to=TensorProto.FLOAT16),
                helper.make_node('Cast', ['half'], ['out'], to=TensorProto.FLOAT),
            ],
            'baked',
            [helper.make_tensor_value_info('images', TensorProto.FLOAT, ['b', 3, 2, 2])],
            [helper.make_tensor_value_info('out', TensorProto.FLOAT, ['b', 3, 2, 2])],
        )
        onnx.save(_model(graph), str(src))
        uint8_input.add_uint8_input(src, dst)
        pixels = np.full((1, 3, 2, 2), 255, dtype=np.uint8)
        np.testing.assert_allclose(_run(dst, pixels), np.ones((1, 3, 2, 2)), atol=1e-3)

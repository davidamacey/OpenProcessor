"""Tests for the half-split rotary layout pass (``export/pe_rope_layout.py``).

The pass rewrites the PE-Core image encoder ONNX so queries and keys use contiguous even/odd
channel halves instead of interleaved pairs. These tests build a one-block attention graph with
the node names of the real export (tiny sizes, ONNX Runtime CPU provider) and check that the
rewritten graph computes the same function.
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

import pe_rope_layout as layout  # noqa: E402


GEOMETRY = layout.Geometry(heads=2, head_dim=8, seq_len=5)
P = '/visual/transformer/resblocks.0/attn/'
WIDTH = GEOMETRY.width


def _ints(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def _build_block(path: Path, seed: int = 0) -> np.ndarray:
    """Write a one-block attention graph with the export's names; returns the QKV weight."""
    rng = np.random.default_rng(seed)
    heads, head_dim, seq = GEOMETRY.heads, GEOMETRY.head_dim, GEOMETRY.seq_len
    weight = rng.standard_normal((WIDTH, 3 * WIDTH)).astype(np.float32) * 0.3
    inits = {
        'qkv_weight': weight,
        'qkv_bias': rng.standard_normal(3 * WIDTH).astype(np.float32) * 0.1,
        'angles': rng.standard_normal((1, 1, seq, head_dim)).astype(np.float32),
        'scale_q': np.array(0.7, dtype=np.float32),
        'scale_k': np.array(0.6, dtype=np.float32),
        'zero': _ints([0]),
        'one': _ints([1]),
        'dim': _ints([head_dim]),
        'axis3': _ints([3]),
        'step2': _ints([2]),
        'last': _ints([-1]),
        'minus1': _ints([-1]),
        'heads_shape': _ints([0, seq, heads, head_dim]),
        'pair_shape': _ints([0, heads, seq, head_dim]),
        'merge_shape': _ints([0, seq, WIDTH]),
    }
    for name, lo, hi in (('q', 0, WIDTH), ('k', WIDTH, 2 * WIDTH), ('v', 2 * WIDTH, 3 * WIDTH)):
        inits[f'{name}_lo'], inits[f'{name}_hi'] = _ints([lo]), _ints([hi])
    nodes = []

    def add(op, inputs, output, name, **attrs):
        nodes.append(helper.make_node(op, inputs, [P + output], name=P + name, **attrs))

    def rope(tag):
        add('Slice', ['qkv', f'{tag}_lo', f'{tag}_hi', 'last'], f'{tag}_cols', f'{tag}_slice')
        add('Reshape', [P + f'{tag}_cols', 'heads_shape'], f'{tag}_split', f'{tag}_reshape')
        add(
            'Transpose', [P + f'{tag}_split'], f'{tag}_heads', f'{tag}_transpose', perm=[0, 2, 1, 3]
        )
        x = P + f'{tag}_heads'
        add('Slice', [x, 'zero', 'dim', 'axis3', 'step2'], f'{tag}_even', f'{tag}_even')
        add('Slice', [x, 'one', 'dim', 'axis3', 'step2'], f'{tag}_odd', f'{tag}_odd')
        add('Neg', [P + f'{tag}_odd'], f'{tag}_neg', f'{tag}_neg')
        add('Unsqueeze', [P + f'{tag}_neg', 'last'], f'{tag}_neg_u', f'{tag}_neg_u')
        add('Unsqueeze', [P + f'{tag}_even', 'last'], f'{tag}_even_u', f'{tag}_even_u')
        add(
            'Concat',
            [P + f'{tag}_neg_u', P + f'{tag}_even_u'],
            f'{tag}_pairs',
            f'{tag}_pairs',
            axis=-1,
        )
        add('Reshape', [P + f'{tag}_pairs', 'pair_shape'], f'{tag}_rot', f'{tag}_rot')
        add('Mul', [x, P + 'Cos_output_0'], f'{tag}_cos', f'{tag}_cos')
        add('Mul', [P + f'{tag}_rot', P + 'Sin_output_0'], f'{tag}_sin', f'{tag}_sin')
        add('Add', [P + f'{tag}_cos', P + f'{tag}_sin'], f'{tag}_roped', f'{tag}_roped')
        return P + f'{tag}_roped'

    nodes.append(
        helper.make_node('MatMul', ['x', 'qkv_weight'], [P + 'MatMul_output_0'], name=P + 'MatMul')
    )
    nodes.append(
        helper.make_node(
            'Add', ['qkv_bias', P + 'MatMul_output_0'], [P + 'Add_output_0'], name=P + 'Add'
        )
    )
    nodes.append(helper.make_node('Cos', ['angles'], [P + 'Cos_output_0'], name=P + 'Cos'))
    nodes.append(helper.make_node('Sin', ['angles'], [P + 'Sin_output_0'], name=P + 'Sin'))
    nodes.append(helper.make_node('Identity', [P + 'Add_output_0'], ['qkv'], name=P + 'qkv_alias'))
    add('Mul', [rope('q'), 'scale_q'], 'Mul_12_output_0', 'Mul_12')
    add('Transpose', [rope('k')], 'k_t', 'k_t', perm=[0, 1, 3, 2])
    add('Mul', [P + 'k_t', 'scale_k'], 'Mul_13_output_0', 'Mul_13')
    add('Slice', ['qkv', 'v_lo', 'v_hi', 'last'], 'v_cols', 'v_slice')
    add('Reshape', [P + 'v_cols', 'heads_shape'], 'v_split', 'v_reshape')
    add('Transpose', [P + 'v_split'], 'Transpose_3_output_0', 'Transpose_3', perm=[0, 2, 1, 3])
    add('MatMul', [P + 'Mul_12_output_0', P + 'Mul_13_output_0'], 'scores', 'MatMul_1')
    add('Softmax', [P + 'scores'], 'probs', 'Softmax', axis=-1)
    add('MatMul', [P + 'probs', P + 'Transpose_3_output_0'], 'context', 'MatMul_2')
    add('Transpose', [P + 'context'], 'context_t', 'Transpose_5', perm=[0, 2, 1, 3])
    nodes.append(
        helper.make_node('Reshape', [P + 'context_t', 'merge_shape'], ['y'], name=P + 'Reshape_16')
    )
    graph = helper.make_graph(
        nodes,
        'block',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['B', seq, WIDTH])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['B', seq, WIDTH])],
        [numpy_helper.from_array(value, name) for name, value in inits.items()],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=10),
        str(path),
    )
    return weight


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return session.run(None, {'x': x})[0]


class TestPermutation:
    def test_evens_then_odds(self) -> None:
        assert layout.head_permutation(8) == [0, 2, 4, 6, 1, 3, 5, 7]

    def test_column_order_permutes_query_and_key_heads_only(self) -> None:
        order = layout._qk_column_order(width=8, heads=2, head_dim=4)
        assert sorted(order) == list(range(24))
        assert order[:8] == [0, 2, 1, 3, 4, 6, 5, 7]
        assert order[8:16] == [8, 10, 9, 11, 12, 14, 13, 15]
        assert order[16:] == list(range(16, 24))


class TestRewrite:
    def test_output_is_unchanged(self, tmp_path: Path) -> None:
        source, target = tmp_path / 'in.onnx', tmp_path / 'out.onnx'
        _build_block(source)
        layout.permute_rope_layout(source, target, GEOMETRY)
        x = (
            np.random.default_rng(1)
            .standard_normal((3, GEOMETRY.seq_len, WIDTH))
            .astype(np.float32)
        )
        np.testing.assert_allclose(_run(target, x), _run(source, x), rtol=1e-5, atol=1e-6)

    def test_pair_arithmetic_is_gone_and_weights_are_permuted(self, tmp_path: Path) -> None:
        source, target = tmp_path / 'in.onnx', tmp_path / 'out.onnx'
        weight = _build_block(source)
        layout.permute_rope_layout(source, target, GEOMETRY)
        model = onnx.load(str(target))
        assert not [n for n in model.graph.node if n.op_type in {'Unsqueeze', 'Cos', 'Sin'}]
        inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        order = layout._qk_column_order(WIDTH, GEOMETRY.heads, GEOMETRY.head_dim)
        np.testing.assert_array_equal(inits['qkv_weight'], weight[:, order])
        assert inits['rope_cos'].shape == (1, 1, GEOMETRY.seq_len, GEOMETRY.head_dim)

    def test_applying_it_twice_is_refused(self, tmp_path: Path) -> None:
        source, once, twice = tmp_path / 'in.onnx', tmp_path / 'once.onnx', tmp_path / 'twice.onnx'
        _build_block(source)
        layout.permute_rope_layout(source, once, GEOMETRY)
        with pytest.raises(ValueError, match='already applied'):
            layout.permute_rope_layout(once, twice, GEOMETRY)

    def test_a_different_geometry_is_refused(self, tmp_path: Path) -> None:
        source = tmp_path / 'in.onnx'
        _build_block(source)
        with pytest.raises(ValueError, match='rotary table shape'):
            layout.permute_rope_layout(source, tmp_path / 'out.onnx', layout.Geometry(2, 8, 7))


class TestApplyIfSupported:
    def test_a_supported_graph_is_rewritten_in_place(self, tmp_path: Path) -> None:
        path, reference = tmp_path / 'pe.onnx', tmp_path / 'reference.onnx'
        _build_block(path)
        reference.write_bytes(path.read_bytes())
        assert layout.apply_if_supported(path, GEOMETRY) is True
        assert not list(tmp_path.glob('*.rope.onnx'))
        x = (
            np.random.default_rng(3)
            .standard_normal((2, GEOMETRY.seq_len, WIDTH))
            .astype(np.float32)
        )
        np.testing.assert_allclose(_run(path, x), _run(reference, x), rtol=1e-5, atol=1e-6)
        assert path.read_bytes() != reference.read_bytes()

    def test_an_unrecognized_graph_is_left_untouched(self, tmp_path: Path) -> None:
        graph = helper.make_graph(
            [helper.make_node('Identity', ['x'], ['y'], name='only')],
            'other',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])],
        )
        path = tmp_path / 'other.onnx'
        onnx.save(helper.make_model(graph, ir_version=10), str(path))
        before = path.read_bytes()
        assert layout.apply_if_supported(path) is False
        assert path.read_bytes() == before
        assert not list(tmp_path.glob('*.rope.onnx'))

"""Rewrite the rotary-embedding layout of the PE-Core image encoder ONNX.

The exported graph applies the rotary angles to interleaved pairs of head channels
(``x0 x1 x2 x3 ...`` rotated as ``(-x1, x0), (-x3, x2) ...``): per block a reshape to a trailing
pair axis, a two-way split on that axis, a negate and a concat. TensorRT lowers that to strided
gathers with a trailing dimension of 2, and the rotary kernels take about a quarter of the engine
time.

Dot products between queries and keys do not change when the same permutation is applied to the
channels of both. This pass therefore permutes the query and key channels of every head so the
pairs become two contiguous halves (all even channels, then all odd ones), by permuting the columns
of the QKV projection weight and bias, and replaces the pair arithmetic with the half-split form
``x * cos + concat(-x[32:], x[:32]) * sin`` using the cosine and sine tables permuted the same way.
Every other value in the graph is unchanged; only floating point rounding differs.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np
    from onnx import GraphProto, NodeProto


logger = logging.getLogger(__name__)

_BLOCK = '/visual/transformer/resblocks.{}/attn/'


@dataclass(frozen=True)
class Geometry:
    """Attention shape of the graph (defaults: PE-Core-L14-336, 24 x 16 heads of 64, 576 + 1 tokens)."""

    heads: int = 16
    head_dim: int = 64
    seq_len: int = 577

    @property
    def width(self) -> int:
        return self.heads * self.head_dim


PE_CORE_L14_336 = Geometry()


def head_permutation(head_dim: int) -> list[int]:
    """Channel order that moves the even channels of a head before the odd ones."""
    return [*range(0, head_dim, 2), *range(1, head_dim, 2)]


def _qk_column_order(width: int, heads: int, head_dim: int) -> list[int]:
    """Column order of a ``[q | k | v]`` projection with q and k permuted per head."""
    perm = head_permutation(head_dim)
    order: list[int] = []
    for section in (0, 1):
        for head in range(heads):
            base = section * width + head * head_dim
            order.extend(base + channel for channel in perm)
    order.extend(range(2 * width, 3 * width))
    return order


def _constant_input(graph: GraphProto, node: NodeProto) -> np.ndarray:
    """The value of the one constant input of ``node`` (an initializer or a Constant node)."""
    from onnx import numpy_helper

    inits = {init.name: init for init in graph.initializer}
    producers = {out: n for n in graph.node for out in n.output}
    for tensor in node.input:
        if tensor in inits:
            return numpy_helper.to_array(inits[tensor])
        source = producers.get(tensor)
        if source is not None and source.op_type == 'Constant':
            return numpy_helper.to_array(source.attribute[0].t)
    raise ValueError(f'{node.name}: no constant input')


def _ancestors(graph: GraphProto, roots: list[str], stop: str, prefix: str) -> set[int]:
    """Indices of the nodes named under ``prefix`` that feed ``roots``, not crossing ``stop``."""
    producer = {out: index for index, node in enumerate(graph.node) for out in node.output}
    seen: set[int] = set()
    todo = list(roots)
    while todo:
        tensor = todo.pop()
        index = producer.get(tensor)
        if tensor == stop or index is None or index in seen:
            continue
        if not graph.node[index].name.startswith(prefix):
            continue
        seen.add(index)
        todo.extend(graph.node[index].input)
    return seen


def _prune_unused(graph: GraphProto) -> None:
    """Drop nodes and initializers that no graph output depends on."""
    producer = {out: node for node in graph.node for out in node.output}
    live: set[int] = set()
    live_tensors: set[str] = set()
    todo = [out.name for out in graph.output]
    while todo:
        tensor = todo.pop()
        if tensor in live_tensors:
            continue
        live_tensors.add(tensor)
        node = producer.get(tensor)
        if node is not None and id(node) not in live:
            live.add(id(node))
            todo.extend(node.input)
    kept_nodes = [node for node in graph.node if id(node) in live]
    kept_inits = [init for init in graph.initializer if init.name in live_tensors]
    del graph.node[:]
    graph.node.extend(kept_nodes)
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)


def rotary_tables(onnx_path: str | Path, block: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the model's own cosine and sine tables (``[1, 1, seq, head_dim]``, float32)."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    names = [f'{_BLOCK.format(block)}Cos_output_0', f'{_BLOCK.format(block)}Sin_output_0']
    for name in names:
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        )
    session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    first = session.get_inputs()[0]
    image = np.zeros([dim if isinstance(dim, int) else 1 for dim in first.shape], dtype=np.float32)
    cos, sin = session.run(names, {first.name: image})
    return cos, sin


def _block_nodes(prefix: str, qkv: str, outputs: tuple[str, str, str]) -> list[NodeProto]:
    """The half-split rotary nodes of one block, producing the three tensors of ``outputs``.

    ``outputs`` are the scaled queries ``[B, H, S, D]``, the scaled keys transposed
    ``[B, H, D, S]`` and the values ``[B, H, S, D]`` that the attention matmuls consume;
    ``qkv`` is the bias-added ``[B, S, 3 * H * D]`` projection.
    """
    from onnx import helper

    scaled_q, scaled_k_t, value = outputs
    nodes: list[NodeProto] = []

    def add(op: str, inputs: list[str], output: str, name: str, **attrs: Any) -> None:
        nodes.append(helper.make_node(op, inputs, [output], name=f'{prefix}rope_{name}', **attrs))

    def head_view(tag: str, out: str | None = None) -> str:
        out = out or f'{prefix}{tag}_heads'
        add(
            'Slice',
            [qkv, f'rope_{tag}_lo', f'rope_{tag}_hi', 'rope_last'],
            f'{prefix}{tag}_cols',
            f'{tag}_slice',
        )
        add(
            'Reshape',
            [f'{prefix}{tag}_cols', 'rope_split'],
            f'{prefix}{tag}_split',
            f'{tag}_reshape',
        )
        add('Transpose', [f'{prefix}{tag}_split'], out, f'{tag}_transpose', perm=[0, 2, 1, 3])
        return out

    def rotate(tag: str, heads: str) -> str:
        def step(op: str, inputs: list[str], suffix: str, **attrs: Any) -> str:
            add(op, inputs, f'{prefix}{tag}_{suffix}', f'{tag}_{suffix}', **attrs)
            return f'{prefix}{tag}_{suffix}'

        low = step('Slice', [heads, 'rope_zero', 'rope_half', 'rope_axis'], 'x1')
        high = step('Slice', [heads, 'rope_half', 'rope_full', 'rope_axis'], 'x2')
        rotated = step('Concat', [step('Neg', [high], 'neg'), low], 'rot', axis=3)
        return step(
            'Add',
            [
                step('Mul', [heads, 'rope_cos'], 'cosmul'),
                step('Mul', [rotated, 'rope_sin'], 'sinmul'),
            ],
            'roped',
        )

    add('Mul', [rotate('q', head_view('q')), f'{prefix}rope_scale_q'], scaled_q, 'q_scale')
    add(
        'Transpose',
        [rotate('k', head_view('k'))],
        f'{prefix}k_t',
        'kt_transpose',
        perm=[0, 1, 3, 2],
    )
    add('Mul', [f'{prefix}k_t', f'{prefix}rope_scale_k'], scaled_k_t, 'k_scale')
    head_view('v', out=value)
    return nodes


def permute_rope_layout(
    onnx_path: str | Path,
    output_path: str | Path,
    geometry: Geometry = PE_CORE_L14_336,
) -> Path:
    """Write ``output_path``: ``onnx_path`` with the half-split rotary layout (see module doc).

    Raises:
        ValueError: The graph is not the expected PE-Core vision transformer (block count, QKV
            weight shape or rotary table shape differ).
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    heads, head_dim, seq_len, width = (
        geometry.heads,
        geometry.head_dim,
        geometry.seq_len,
        geometry.width,
    )
    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    if 'rope_cos' in inits:
        raise ValueError(f'{onnx_path}: the half-split rotary layout is already applied')
    by_name = {node.name: node for node in graph.node}
    if not all(_BLOCK.format(0) + name in by_name for name in ('MatMul', 'Cos', 'Sin')):
        raise ValueError(
            f'{onnx_path}: not a PE-Core vision transformer (no rotary attention block)'
        )
    cos, sin = rotary_tables(onnx_path)
    if cos.shape != (1, 1, seq_len, head_dim) or sin.shape != cos.shape:
        raise ValueError(f'unexpected rotary table shape {cos.shape}')
    perm = head_permutation(head_dim)
    cos_p = np.ascontiguousarray(cos[..., perm])
    sin_p = np.ascontiguousarray(sin[..., perm])

    order = _qk_column_order(width, heads, head_dim)

    consts = {
        'rope_cos': cos_p,
        'rope_sin': sin_p,
        'rope_half': np.array([head_dim // 2], dtype=np.int64),
        'rope_full': np.array([head_dim], dtype=np.int64),
        'rope_zero': np.array([0], dtype=np.int64),
        'rope_axis': np.array([3], dtype=np.int64),
        'rope_split': np.array([0, seq_len, heads, head_dim], dtype=np.int64),
        'rope_q_lo': np.array([0], dtype=np.int64),
        'rope_k_lo': np.array([width], dtype=np.int64),
        'rope_v_lo': np.array([2 * width], dtype=np.int64),
        'rope_q_hi': np.array([width], dtype=np.int64),
        'rope_k_hi': np.array([2 * width], dtype=np.int64),
        'rope_v_hi': np.array([3 * width], dtype=np.int64),
        'rope_last': np.array([-1], dtype=np.int64),
    }
    for name, value in consts.items():
        graph.initializer.append(numpy_helper.from_array(value, name))

    blocks = 0
    after: dict[int, list] = {}
    drop: set[int] = set()
    index_of = {id(node): index for index, node in enumerate(graph.node)}
    while (prefix := _BLOCK.format(blocks)) + 'MatMul' in by_name:
        missing = [
            name
            for name in ('MatMul', 'Add', 'Mul_12', 'Mul_13', 'Transpose_3')
            if prefix + name not in by_name
        ]
        if missing:
            raise ValueError(f'block {blocks}: expected nodes not found: {missing}')
        matmul = by_name[prefix + 'MatMul']
        bias_add = by_name[prefix + 'Add']
        weight = numpy_helper.to_array(inits[matmul.input[1]])
        if weight.shape != (width, 3 * width):
            raise ValueError(f'block {blocks}: unexpected QKV weight shape {weight.shape}')
        bias_name = next(i for i in bias_add.input if i in inits)
        bias = numpy_helper.to_array(inits[bias_name])
        weight, bias = weight[:, order], bias[order]
        inits[matmul.input[1]].CopyFrom(
            numpy_helper.from_array(np.ascontiguousarray(weight), matmul.input[1])
        )
        inits[bias_name].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(bias), bias_name))

        qkv = bias_add.output[0]
        scaled_q, scaled_k_t, value_name = (
            prefix + 'Mul_12_output_0',
            prefix + 'Mul_13_output_0',
            prefix + 'Transpose_3_output_0',
        )
        drop |= {
            index_of[id(graph.node[i])]
            for i in _ancestors(graph, [scaled_q, scaled_k_t, value_name], qkv, prefix)
        }

        # The scale constants of the original graph (Mul_12 / Mul_13) are kept by value.
        for suffix, mul_name in (('scale_q', 'Mul_12'), ('scale_k', 'Mul_13')):
            scale = _constant_input(graph, by_name[prefix + mul_name])
            graph.initializer.append(numpy_helper.from_array(scale, f'{prefix}rope_{suffix}'))

        after.setdefault(id(bias_add), []).extend(
            _block_nodes(prefix, qkv, (scaled_q, scaled_k_t, value_name))
        )
        blocks += 1

    if blocks == 0:
        raise ValueError('no attention blocks found')
    ordered: list = []
    for index, node in enumerate(graph.node):
        if index not in drop:
            ordered.append(node)
            ordered.extend(after.get(id(node), []))
    del graph.node[:]
    graph.node.extend(ordered)
    _prune_unused(graph)
    onnx.checker.check_model(model, full_check=False)
    out = Path(output_path)
    onnx.save(model, str(out))
    return out


def apply_if_supported(onnx_path: str | Path, geometry: Geometry = PE_CORE_L14_336) -> bool:
    """Rewrite ``onnx_path`` in place; leave it untouched (and say so) if the graph is not PE-Core's.

    The rewritten graph computes the same function, so an unsupported export only costs speed.
    """
    path = Path(onnx_path)
    staged = path.with_suffix('.rope.onnx')
    try:
        permute_rope_layout(path, staged, geometry)
    except ValueError as exc:
        staged.unlink(missing_ok=True)
        logger.warning('rotary layout pass skipped, the engine will be slower: %s', exc)
        return False
    staged.replace(path)
    logger.info('rotary layout pass applied: %s', path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(permute_rope_layout(args.input, args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

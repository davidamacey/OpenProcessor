"""Graph edits for the frame engine's OCR builds (ONNX, before the TensorRT build).

The recogniser's output is ``[B, T, classes]`` probabilities (18,710 classes): shipping that to the
engine's host memory is what made big padded batches slow, and CTC decoding only needs each time
step's best class and its probability. :func:`add_argmax_outputs` makes the graph end in those two
small tensors.
"""

from __future__ import annotations

from pathlib import Path


INDICES = 'indices'
SCORES = 'scores'


def add_argmax_outputs(onnx_path: str | Path, output_path: str | Path) -> Path:
    """Write ``output_path``: ``onnx_path``'s only output ``[B, T, C]`` replaced by ``indices``
    (``int32 [B, T]``, ``ArgMax`` over the last axis) and ``scores`` (``float32 [B, T]``,
    ``ReduceMax``). ``ArgMax`` returns the first maximal index, as ``numpy.argmax`` does.

    Raises:
        ValueError: the graph does not have exactly one output.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    if len(graph.output) != 1:
        raise ValueError(f'{onnx_path}: expected one output, got {[o.name for o in graph.output]}')
    old = graph.output[0]
    wide = f'{old.name}__argmax_i64'
    graph.node.extend(
        [
            helper.make_node('ArgMax', [old.name], [wide], axis=2, keepdims=0),
            helper.make_node('Cast', [wide], [INDICES], to=TensorProto.INT32),
            helper.make_node('ReduceMax', [old.name], [SCORES], axes=[2], keepdims=0),
        ]
    )
    dims = [
        d.dim_value if d.dim_value else d.dim_param or None for d in old.type.tensor_type.shape.dim
    ]
    graph.output.remove(old)
    graph.output.extend(
        [
            helper.make_tensor_value_info(INDICES, TensorProto.INT32, dims[:2]),
            helper.make_tensor_value_info(SCORES, TensorProto.FLOAT, dims[:2]),
        ]
    )
    out = Path(output_path)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(out))
    return out

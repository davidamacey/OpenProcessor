"""Turn a float image input of an ONNX graph into a uint8 input with the scaling in the graph.

Sending ``uint8`` pixels to Triton and letting the engine do ``(v / 255 - mean) / std`` moves
4x fewer bytes over gRPC than the float tensor the client used to build. The prefix is the same
arithmetic the client's lookup table used (divide, subtract, divide, in float32), placed in front
of every consumer of the input, so the rest of the graph, and an FP16 bake that ran before this,
are untouched. TensorRT fuses the cast and the arithmetic into the first layer.

Usage as a script (for build scripts that shell out to ``trtexec``)::

    python export/uint8_input.py IN.onnx OUT.onnx [--mean M M M --std S S S]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_uint8_input(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> Path:
    """Write ``output_path``: ``onnx_path`` with a uint8 NCHW input normalized in the graph.

    The graph must have exactly one input and it must be float32 (the exporters keep the I/O
    float32 even after an FP16 bake). The input keeps its name and shape (including a dynamic
    batch). With ``mean`` and ``std`` the pixel becomes ``(v / 255 - mean) / std`` per channel;
    without them it becomes ``v / 255``.

    Raises:
        ValueError: The graph does not have a single float32 input (so it was already converted),
            or only one of ``mean`` / ``std`` was given.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if (mean is None) != (std is None):
        raise ValueError('mean and std go together')
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    real_inputs = [i for i in graph.input if i.name not in initializer_names]
    if len(real_inputs) != 1 or real_inputs[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError(
            f'{onnx_path}: expected one float32 image input, got '
            f'{[(i.name, i.type.tensor_type.elem_type) for i in real_inputs]}'
        )
    old = real_inputs[0]
    name = old.name
    normalized = f'{name}__normalized'

    for node in graph.node:
        for index, tensor in enumerate(node.input):
            if tensor == name:
                node.input[index] = normalized

    def constant(suffix: str, values: object) -> str:
        const_name = f'{name}__{suffix}'
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(values, dtype=np.float32), const_name)
        )
        return const_name

    steps = [
        helper.make_node('Cast', [name], [f'{name}__f32'], to=TensorProto.FLOAT),
        helper.make_node('Div', [f'{name}__f32', constant('scale', 255.0)], [f'{name}__unit']),
    ]
    last = f'{name}__unit'
    if mean is not None and std is not None:
        steps.append(
            helper.make_node(
                'Sub', [last, constant('mean', np.reshape(mean, (1, 3, 1, 1)))], [f'{name}__ctr']
            )
        )
        steps.append(
            helper.make_node(
                'Div',
                [f'{name}__ctr', constant('std', np.reshape(std, (1, 3, 1, 1)))],
                [normalized],
            )
        )
    else:
        steps[-1].output[0] = normalized
    for position, node in enumerate(steps):
        graph.node.insert(position, node)

    uint8 = helper.make_tensor_value_info(name, TensorProto.UINT8, None)
    uint8.type.tensor_type.shape.CopyFrom(old.type.tensor_type.shape)
    graph.input.remove(old)
    graph.input.insert(0, uint8)

    out = Path(output_path)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(out))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('onnx_in')
    parser.add_argument('onnx_out')
    parser.add_argument('--mean', type=float, nargs=3)
    parser.add_argument('--std', type=float, nargs=3)
    args = parser.parse_args(argv)
    mean = tuple(args.mean) if args.mean else None
    std = tuple(args.std) if args.std else None
    print(add_uint8_input(args.onnx_in, args.onnx_out, mean=mean, std=std))  # type: ignore[arg-type]
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

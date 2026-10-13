"""Fail when an ONNX about to be built into a TensorRT engine disagrees with its Triton config.

Triton refuses to load a plan whose input datatype differs from ``config.pbtxt`` ("unexpected
datatype TYPE_FP32 for inference input ... expecting TYPE_UINT8"). Builds are strongly typed, so
the plan input is the ONNX input; checking the ONNX at build time turns that load failure into an
install failure.

Usage::

    python export/check_input_dtype.py MODEL.onnx config.pbtxt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


_ONNX_TO_TRITON = {1: 'TYPE_FP32', 2: 'TYPE_UINT8', 3: 'TYPE_INT8', 6: 'TYPE_INT32', 7: 'TYPE_INT64', 10: 'TYPE_FP16'}


def config_input_dtype(config_path: str | Path) -> str:
    """The ``data_type`` of the first entry of the ``input [ ... ]`` block of a config.pbtxt."""
    text = Path(config_path).read_text()
    block = re.search(r'^input\s*\[(.*?)^\]', text, re.DOTALL | re.MULTILINE)
    match = re.search(r'data_type:\s*(TYPE_\w+)', block.group(1)) if block else None
    if match is None:
        raise ValueError(f'{config_path}: no input data_type found')
    return match.group(1)


def onnx_input_dtype(onnx_path: str | Path) -> str:
    """The Triton name of the datatype of the single non-initializer input of an ONNX."""
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    weights = {init.name for init in graph.initializer}
    inputs = [i for i in graph.input if i.name not in weights]
    if len(inputs) != 1:
        raise ValueError(f'{onnx_path}: expected one input, got {[i.name for i in inputs]}')
    elem = inputs[0].type.tensor_type.elem_type
    return _ONNX_TO_TRITON.get(elem, f'onnx_elem_type_{elem}')


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    built, wanted = onnx_input_dtype(args[0]), config_input_dtype(args[1])
    if built != wanted:
        print(
            f'[ERROR] input datatype mismatch: {args[0]} has {built}, {args[1]} declares {wanted}. '
            'Triton would refuse to load this engine.',
            file=sys.stderr,
        )
        return 1
    print(f'input datatype ok: {built}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""nd:P0 product-engine vectors without Triton: the product's TensorRT plan run directly (TensorRT Python API).

The plan files (``models/<name>/1/model.plan``) are the product's engines; the preprocessing is the product's
own function. Run in a venv that has ``tensorrt-cu13`` (pinned to the engine's TensorRT version, see
``requirements-export.txt``) and ``torch`` for device buffers; it writes one ``.npy`` with unit-norm rows in
the order of the path list, to be compared with ``encode.py`` vectors through ``parity.py compare``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def run_plan(plan: Path, inputs: np.ndarray, batch: int = 8) -> np.ndarray:
    import tensorrt as trt
    import torch

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(plan.read_bytes())
    if engine is None:
        raise SystemExit(f'cannot load {plan}')
    ctx = engine.create_execution_context()
    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    in_dtype = {trt.DataType.UINT8: torch.uint8, trt.DataType.FLOAT: torch.float32}[
        engine.get_tensor_dtype(in_name)
    ]
    stream = torch.cuda.Stream()
    outs = []
    for start in range(0, len(inputs), batch):
        chunk = torch.from_numpy(inputs[start : start + batch]).cuda().to(in_dtype)
        if in_dtype is torch.float32:
            chunk = chunk / 255.0
        chunk = chunk.contiguous()
        ctx.set_input_shape(in_name, tuple(chunk.shape))
        out = torch.empty((chunk.shape[0], *engine.get_tensor_shape(out_name)[1:]), device='cuda')
        ctx.set_tensor_address(in_name, chunk.data_ptr())
        ctx.set_tensor_address(out_name, out.data_ptr())
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        outs.append(torch.nn.functional.normalize(out, dim=-1).cpu().numpy())
    return np.concatenate(outs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paths', type=Path, required=True)
    ap.add_argument('--space', choices=['pe_frame', 'mobileclip_frame'], required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    from scripts.research.near_dup import encode

    paths = [ln for ln in args.paths.read_text().splitlines() if ln]
    x = np.concatenate(list(encode.batches(paths, args.space, 32, 4)))
    # the PE plan takes uint8 CHW; the MobileCLIP plan takes float CHW in [0, 1]
    chw = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    np.save(args.out, run_plan(args.plan, chw))
    sys.stdout.write(f'{len(paths)} product-engine vectors -> {args.out}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

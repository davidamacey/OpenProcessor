"""A small TensorRT runner over CuPy device memory for the shipped plans (read-only).

TensorRT Python API: ``Runtime.deserialize_cuda_engine``, ``IExecutionContext.set_input_shape``,
``set_tensor_address`` and ``execute_async_v3`` (NVIDIA TensorRT developer guide, "Python API").
A pool of execution contexts, each with its own CUDA stream, stands in for the instance and dynamic
batcher of the Triton model repository. Plans that use ``EfficientNMS_TRT`` need the plugin library
registered first.
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
import tensorrt as trt


if TYPE_CHECKING:
    from pathlib import Path


_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.UINT8: np.uint8,
}


class _Ctx:
    def __init__(self, engine: Any) -> None:
        self.context = engine.create_execution_context()
        # no per-layer NVTX ranges, so an nsys trace attributes the kernels to the caller's stage range
        setattr(self.context, 'nvtx_verbosity', trt.ProfilingVerbosity.NONE)  # noqa: B010  # attribute write only; static analysis sees no reader
        self.stream = cp.cuda.Stream(non_blocking=True)


class TrtModel:
    """``run(batch) -> {output name: numpy array}`` for a single dynamic-batch input."""

    def __init__(self, plan: Path, contexts: int = 4) -> None:
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, '')
        engine = trt.Runtime(logger).deserialize_cuda_engine(plan.read_bytes())
        if engine is None:
            raise RuntimeError(f'could not deserialize {plan.name}')
        self.engine = engine
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        io = engine.get_tensor_mode
        self.input = next(n for n in names if io(n) == trt.TensorIOMode.INPUT)
        self.outputs = [n for n in names if io(n) == trt.TensorIOMode.OUTPUT]
        self.max_batch = int(engine.get_tensor_profile_shape(self.input, 0)[2][0])
        self.pool: queue.Queue[_Ctx] = queue.Queue()
        for _ in range(contexts):
            self.pool.put(_Ctx(engine))

    def run(self, batch: Any) -> dict[str, np.ndarray]:
        """``batch``: a contiguous CuPy array of the input dtype; at most ``max_batch`` rows."""
        ctx = self.pool.get()
        try:
            produced = cp.cuda.Event()
            produced.record(cp.cuda.get_current_stream())  # the caller's kernels wrote the batch
            if not ctx.context.set_input_shape(self.input, tuple(batch.shape)):
                raise ValueError(f'batch shape {tuple(batch.shape)} is outside the plan profile')
            with ctx.stream:
                ctx.stream.wait_event(produced)
                outs = {
                    name: cp.empty(
                        tuple(ctx.context.get_tensor_shape(name)),
                        _NP[self.engine.get_tensor_dtype(name)],
                    )
                    for name in self.outputs
                }
                ctx.context.set_tensor_address(self.input, int(batch.data.ptr))
                for name, arr in outs.items():
                    ctx.context.set_tensor_address(name, int(arr.data.ptr))
                if not ctx.context.execute_async_v3(ctx.stream.ptr):
                    raise RuntimeError('TensorRT execute_async_v3 failed')
                ctx.stream.synchronize()
                return {k: v.get() for k, v in outs.items()}
        finally:
            self.pool.put(ctx)

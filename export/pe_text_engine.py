"""PE text tower TensorRT plan: profile, build, parity gate (the gate and profile are pure).

The plan is built the way every other FP16 engine here is: FP16 is baked into the ONNX
(``trt_utils.bake_fp16_onnx``; TensorRT 11 is strongly typed and has no FP16 flag), the ONNX is parsed
and built with the TensorRT Python API, and the result is validated and written atomically
(``trt_utils.atomic_write_plan``). Token ids stay INT64 in the graph interface; TensorRT casts them itself.

Reference docs: TensorRT 11.4 developer guide, "Working with Dynamic Shapes" (optimization profiles);
Triton model configuration (``tensorrt_plan``, ``max_batch_size``, dynamic batching).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pe_text_constants import (
    CONTEXT_LENGTH,
    ENGINE_MIN_TOKENS,
    ENGINE_OPT_BATCH,
    ENGINE_OPT_TOKENS,
    ENGINE_PARITY_MEDIAN_MIN,
    ENGINE_PARITY_P01_MIN,
    INPUT_TENSOR,
    OUTPUT_TENSOR,
    PARITY_PROMPTS,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


logger = logging.getLogger(__name__)


def profile_shapes(max_batch: int) -> dict[str, tuple[int, int]]:
    """``(batch, tokens)`` of the min / opt / max optimization profile."""
    if max_batch < ENGINE_OPT_BATCH:
        raise ValueError(f'max_batch must be >= {ENGINE_OPT_BATCH}, got {max_batch}')
    return {
        'min': (1, ENGINE_MIN_TOKENS),
        'opt': (ENGINE_OPT_BATCH, ENGINE_OPT_TOKENS),
        'max': (max_batch, CONTEXT_LENGTH),
    }


def gate_prompts() -> list[str]:
    """Distinct prompts over many token lengths: the parity prompts and every ordered pair joined."""
    base = list(PARITY_PROMPTS)
    pairs = [f'{a}, {b}' for a in base for b in base if a is not b]
    return list(dict.fromkeys(base + pairs))


@dataclass(frozen=True)
class GateReport:
    """Row-wise cosine agreement of the engine with the FP32 reference."""

    n: int
    median: float
    p01: float
    minimum: float

    def passes(
        self, median_min: float = ENGINE_PARITY_MEDIAN_MIN, p01_min: float = ENGINE_PARITY_P01_MIN
    ) -> bool:
        return self.n > 0 and self.median >= median_min and self.p01 >= p01_min


def compare(reference: Any, candidate: Any) -> GateReport:
    """Cosine of each row of ``candidate`` against ``reference`` (``[N, D]`` each)."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape or ref.ndim != 2 or ref.shape[0] == 0:
        raise ValueError(f'shape mismatch: reference {ref.shape} vs candidate {cand.shape}')
    cos = (ref * cand).sum(axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1))
    if not np.isfinite(cos).all():
        return GateReport(n=len(cos), median=float('nan'), p01=float('nan'), minimum=float('nan'))
    return GateReport(
        n=len(cos),
        median=float(np.median(cos)),
        p01=float(np.percentile(cos, 1)),
        minimum=float(cos.min()),
    )


class EngineGateError(RuntimeError):
    """The built engine does not match the FP32 graph; it must not be installed."""


def build_and_gate(
    build: Callable[[], bytes],
    run_engine: Callable[[bytes, Sequence[Any]], list[Any]],
    reference: Callable[[Any], Any],
    token_rows: Sequence[Any],
    install: Callable[[bytes], Path],
) -> GateReport:
    """Build the plan, compare it with the reference row by row, install only when the gate passes.

    ``run_engine(plan, rows)`` returns one ``[1, D]`` embedding per ``[1, T]`` token row.
    """
    import numpy as np

    plan = build()
    cand = np.concatenate(run_engine(plan, token_rows), axis=0)
    ref = np.concatenate([reference(row) for row in token_rows], axis=0)
    report = compare(ref, cand)
    logger.info(
        f'Engine parity vs FP32: n={report.n} median={report.median:.6f} '
        f'p01={report.p01:.6f} min={report.minimum:.6f}'
    )
    if not report.passes():
        raise EngineGateError(
            f'engine parity failed: median {report.median:.6f} (>= {ENGINE_PARITY_MEDIAN_MIN}), '
            f'p01 {report.p01:.6f} (>= {ENGINE_PARITY_P01_MIN}); not installing'
        )
    install(plan)
    return report


def build_plan_bytes(onnx_path: Path, max_batch: int, workspace_bytes: int) -> bytes:
    """Serialize an FP16 plan for the text ONNX (needs TensorRT and a GPU; exercised by the wave)."""
    import tensorrt as trt
    from trt_utils import bake_fp16_onnx, create_explicit_network

    fp16_onnx = bake_fp16_onnx(onnx_path)
    if fp16_onnx == onnx_path:
        raise RuntimeError('FP16 bake returned the FP32 graph; refusing an FP32 engine')
    logger_trt = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger_trt)
    network = create_explicit_network(builder)
    parser = trt.OnnxParser(network, logger_trt)
    if not parser.parse(fp16_onnx.read_bytes(), str(fp16_onnx)):
        errors = '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'TensorRT could not parse {fp16_onnx}: {errors}')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    shapes = profile_shapes(max_batch)
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_TENSOR, shapes['min'], shapes['opt'], shapes['max'])
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT failed to build the PE text engine')
    return bytes(plan)


def run_plan(plan: bytes, rows: Sequence[Any]) -> list[Any]:
    """Run ``plan`` on each ``[1, T]`` int64 row through Polygraphy's TrtRunner (no CUDA bindings needed)."""
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner

    with TrtRunner(EngineFromBytes(plan)) as runner:
        return [runner.infer({INPUT_TENSOR: row})[OUTPUT_TENSOR] for row in rows]

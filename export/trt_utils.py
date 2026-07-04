"""Shared TensorRT builder helpers for the export scripts.

Centralizes the version-sensitive bits of engine building so the per-model
export scripts don't each carry their own compatibility branches.
"""

from __future__ import annotations

import tensorrt as trt


def create_explicit_network(builder: trt.Builder) -> trt.INetworkDefinition:
    """Create an explicit-batch network across TensorRT versions.

    TRT >= 10 networks are always explicit-batch; the
    ``NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` flag was deprecated in
    10.x and removed in newer majors. Pass it only where it still exists.
    """
    flag = getattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH', None)
    if flag is not None:
        return builder.create_network(1 << int(flag))
    return builder.create_network(0)


def enable_fp16(builder: trt.Builder, config: trt.IBuilderConfig) -> bool:
    """Request an FP16 engine build where the builder still supports it.

    TensorRT 11 is strongly-typed only: ``BuilderFlag.FP16`` (and
    ``Builder.platform_has_fast_fp16``) were removed, and reduced
    precision must instead be baked into the ONNX graph before parsing
    (e.g. NVIDIA ModelOpt AutoCast — see ultralytics' onnx2engine).
    On such builds this returns False and the engine follows the ONNX
    dtypes (FP32 weights run with TF32 tensor cores on Ampere+).

    Never gate precision on ``torch.cuda`` — the export venvs ship
    CPU-only torch while the engine build targets the GPU through
    TensorRT itself.
    """
    fp16_flag = getattr(trt.BuilderFlag, 'FP16', None)
    if fp16_flag is None:
        return False
    if getattr(builder, 'platform_has_fast_fp16', True):
        config.set_flag(fp16_flag)
        return True
    return False

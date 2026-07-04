"""Shared TensorRT builder helpers for the export scripts.

Centralizes the version-sensitive bits of engine building so the per-model
export scripts don't each carry their own compatibility branches.
"""

from __future__ import annotations

from pathlib import Path

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


def engine_output_dtypes(plan_path: str | Path) -> dict[str, str]:
    """Map a serialized engine's output tensors to Triton dtype strings.

    Output precisions are decided by TensorRT (e.g. the EfficientNMS_TRT
    plugin emits FP16 boxes under TRT 11.0 but FP32 under 11.1), so
    config.pbtxt must be written from the BUILT engine rather than
    assumed — a hardcoded dtype breaks across TRT releases.
    """
    dtype_map = {
        trt.DataType.FLOAT: 'TYPE_FP32',
        trt.DataType.HALF: 'TYPE_FP16',
        trt.DataType.INT32: 'TYPE_INT32',
        trt.DataType.INT64: 'TYPE_INT64',
        trt.DataType.BOOL: 'TYPE_BOOL',
        trt.DataType.INT8: 'TYPE_INT8',
    }
    trt_logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(trt_logger, '')
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
    if engine is None:
        raise RuntimeError(f'could not deserialize engine {plan_path}')

    outputs: dict[str, str] = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            outputs[name] = dtype_map.get(engine.get_tensor_dtype(name), 'TYPE_FP32')
    return outputs


def bake_fp16_onnx(onnx_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Bake FP16 mixed precision into an ONNX for TRT >= 11 typed builds.

    TensorRT 11 removed the FP16 builder flag; reduced precision must live
    in the ONNX graph itself. Converts weights/activations to FP16 via
    onnxconverter-common's graph rewrite (no ONNX Runtime execution — it
    also works on graphs containing TRT plugin ops like EfficientNMS_TRT,
    which ORT-based tools cannot type-infer). ``keep_io_types=True`` keeps
    graph inputs/outputs FP32, so Triton config dtypes (TYPE_FP32) remain
    valid and clients are unaffected.

    On older TRT (classic FP16 flag still present) this is a no-op and
    returns the input path — :func:`enable_fp16` handles precision there.
    """
    if getattr(trt.BuilderFlag, 'FP16', None) is not None:
        return Path(onnx_path)

    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(onnx_path))
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        # Numerically sensitive defaults plus ops TRT executes outside the
        # FP16 path anyway (plugins, index math).
        op_block_list=[
            *float16.DEFAULT_OP_BLOCK_LIST,
            'EfficientNMS_TRT',
            'NonMaxSuppression',
            'Range',
        ],
    )
    out = Path(output_path) if output_path else Path(onnx_path).with_suffix('.fp16.onnx')
    onnx.save(fp16_model, str(out))
    return out

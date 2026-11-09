"""The TensorRT engine of a fused detector ONNX: FP16 bake, uint8 input, bounded batch profile."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import tensorrt as trt
from trt_build_settings import workspace_bytes
from trt_utils import atomic_write_plan, create_explicit_network
from uint8_input import add_uint8_input
from yolo_export_settings import IMG_SIZE


logger = logging.getLogger(__name__)


def setup_trt_builder() -> tuple[
    trt.Builder, trt.IBuilderConfig, trt.INetworkDefinition, trt.Logger
]:
    """A TensorRT builder, config (workspace from trt_build_settings) and explicit network."""
    # The ONNX-export step always passes device='cpu' to Ultralytics
    # (yolo_export_settings.DEVICE). Ultralytics' own select_device('cpu')
    # sets os.environ['CUDA_VISIBLE_DEVICES'] = '-1' as a side effect,
    # in-process -- CUDA_VISIBLE_DEVICES is read by the driver, not by torch,
    # so it blinds the TensorRT builder below too (createInferBuilder fails
    # with a CUDA initialization error), even though a bare tensorrt.Builder()
    # in an unpoisoned process works fine. The engine build never uses torch's
    # CUDA, so clearing the poisoned value here is safe and restores what this
    # process would see without Ultralytics having touched it.
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
        logger.warning(
            "Clearing CUDA_VISIBLE_DEVICES=-1 (set by Ultralytics' CPU-mode "
            'ONNX export in this process) before building the TensorRT engine'
        )
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes('yolo'))
    return builder, config, create_explicit_network(builder), trt_logger


def _bake_inputs(onnx_path: Path, max_batch: int) -> tuple[Path, list[Path]]:
    """The ONNX the builder parses, and the intermediate files to remove after the build.

    TensorRT 11 is strongly typed (no FP16 builder flag), so FP16 is baked into the ONNX first
    by NVIDIA ModelOpt AutoCast through the Ultralytics wrapper, which feeds proper calibration
    shapes for the dynamic axes (the plain graph rewrite of ``trt_utils.bake_fp16_onnx`` mis-types
    this graph's Concat nodes). The engine then takes uint8 pixels and scales them itself, a
    quarter of the bytes over gRPC.
    """
    from ultralytics.utils.export.engine import modelopt_quantize_onnx

    fp16_onnx = Path(
        modelopt_quantize_onnx(
            str(onnx_path), quantize=16, shape=(max_batch, 3, IMG_SIZE, IMG_SIZE), dynamic=True
        )
    )
    u8_onnx = add_uint8_input(fp16_onnx, fp16_onnx.with_suffix('.u8.onnx'))
    return u8_onnx, [fp16_onnx, u8_onnx]


def build_fused_engine(onnx_path: Path, plan_path: Path, max_batch: int) -> bool:
    """Build the FP16, uint8-input engine of ``onnx_path`` at ``plan_path``; False when it fails.

    The optimization profile bounds EVERY dynamic axis (batch 1..``max_batch``, fixed 640x640):
    the Ultralytics export leaves H and W dynamic, and an unbounded spatial axis makes TensorRT
    budget for gigantic activations (12+ GB tactics) that fail on consumer GPUs. ``max_batch``
    is the batch the ONNX was exported with (the NMS is unrolled for it).
    """
    u8_onnx, intermediates = _bake_inputs(onnx_path, max_batch)
    try:
        builder, config, network, trt_logger = setup_trt_builder()
        parser = trt.OnnxParser(network, trt_logger)
        if not parser.parse_from_file(str(u8_onnx)):
            for i in range(parser.num_errors):
                logger.error(f'  ONNX parse error [{i}]: {parser.get_error(i)}')
            return False

        profile = builder.create_optimization_profile()
        side = (3, IMG_SIZE, IMG_SIZE)
        shapes = {
            'min': (1, *side),
            'opt': (max(1, max_batch // 2), *side),
            'max': (max_batch, *side),
        }
        for i in range(network.num_inputs):
            profile.set_shape(network.get_input(i).name, **shapes)
        config.add_optimization_profile(profile)

        gc.collect()
        logger.info('Building TensorRT engine (this may take several minutes)...')
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            logger.error('Engine build failed - builder returned None')
            return False
        atomic_write_plan(bytes(serialized), plan_path)
    finally:
        for path in intermediates:
            path.unlink(missing_ok=True)
    logger.info(f'Engine saved: {plan_path} ({plan_path.stat().st_size / 1e6:.1f} MB)')
    return True

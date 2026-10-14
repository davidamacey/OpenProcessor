"""TensorRT builder helpers for the YOLO11 exporter: builder setup, optimization profile, ONNX parse and engine build."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import tensorrt as trt
import torch
from trt_utils import bake_fp16_onnx, create_explicit_network, enable_fp16
from uint8_input import add_uint8_input
from yolo_export_settings import IMG_SIZE, WORKSPACE_GB


logger = logging.getLogger(__name__)


def setup_trt_builder(
    workspace_gb: int = WORKSPACE_GB,
) -> tuple[trt.Builder, trt.IBuilderConfig, trt.INetworkDefinition, trt.Logger]:
    """
    Set up TensorRT builder with common configuration.

    Args:
        workspace_gb: Workspace size in GB

    Returns:
        Tuple of (builder, config, network, logger)
    """
    # This script runs
    # under /opt/venv-y11's deliberately CPU-only torch (see
    # requirements-export-y11.txt), so the ONNX-export step above always
    # passes device='cpu' to Ultralytics. Ultralytics' own
    # select_device('cpu') sets os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # as a side effect, in-process -- CUDA_VISIBLE_DEVICES is read by the
    # driver, not by torch, so it blinds the TensorRT builder below too
    # (createInferBuilder fails with a CUDA initialization error), even
    # though a bare tensorrt.Builder() in an unpoisoned process works
    # fine. The engine build never uses torch's CUDA (see that same
    # requirements file's own comment), so clearing the poisoned value
    # here is safe and restores what this process would see without
    # Ultralytics having touched it.
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

    # Set workspace size (TRT >= 10 memory-pool API; requirements pin TRT 11)
    workspace_bytes = int(workspace_gb * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    network = create_explicit_network(builder)

    return builder, config, network, trt_logger


def add_optimization_profile(
    builder: trt.Builder,
    config: trt.IBuilderConfig,
    network: trt.INetworkDefinition,
    max_batch: int,
) -> None:
    """Add optimization profile for dynamic batching."""
    profile = builder.create_optimization_profile()

    min_shape = (1, 3, IMG_SIZE, IMG_SIZE)
    opt_shape = (max_batch // 2, 3, IMG_SIZE, IMG_SIZE)
    max_shape = (max_batch, 3, IMG_SIZE, IMG_SIZE)

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        profile.set_shape(inp.name, min=min_shape, opt=opt_shape, max=max_shape)
        logger.debug(f'  {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}')

    config.add_optimization_profile(profile)


def enable_fp16_if_available(builder: trt.Builder, config: trt.IBuilderConfig) -> bool:
    """Enable FP16 precision where supported (see trt_utils.enable_fp16)."""
    if enable_fp16(builder, config):
        logger.info('FP16 precision enabled')
        return True
    logger.info(
        'Building typed precision (TRT 11 strongly-typed: FP32 weights + TF32 '
        'tensor cores; bake FP16 into the ONNX via ModelOpt AutoCast to go faster)'
    )
    return False


def parse_onnx_model(
    parser: trt.OnnxParser,
    onnx_path: Path,
    *,
    uint8_input: bool = False,
) -> bool:
    """Parse ONNX model and report any errors.

    ``uint8_input`` puts the ``/255`` scaling in the graph so the engine takes uint8 pixels
    (the FP16 bake runs first and keeps the I/O float32; see ``uint8_input.py``).
    """
    logger.info('Baking FP16 (ModelOpt AutoCast, TRT 11 typed builds)...')
    onnx_path = bake_fp16_onnx(onnx_path)
    if uint8_input:
        onnx_path = add_uint8_input(onnx_path, Path(onnx_path).with_suffix('.u8.onnx'))
    if not parser.parse_from_file(str(onnx_path)):
        logger.error('Failed to parse ONNX model:')
        for i in range(parser.num_errors):
            error = parser.get_error(i)
            logger.error(f'  [{i}] {error}')
        return False
    return True


def build_and_save_engine(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    config: trt.IBuilderConfig,
    output_path: Path,
) -> bool:
    """Build TensorRT engine and save to file."""
    # Free memory before building
    gc.collect()
    torch.cuda.empty_cache()

    logger.info('Building TensorRT engine (this may take 5-10 minutes)...')

    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error('Failed to build engine - builder returned None')
            return False
        with Path(output_path).open('wb') as f:
            f.write(serialized_engine)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f'Engine saved: {output_path} ({file_size_mb:.2f} MB)')
        return True

    except Exception as e:
        logger.error(f'Engine build failed: {e}')
        return False

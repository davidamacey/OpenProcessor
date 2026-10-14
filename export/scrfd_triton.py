"""SCRFD TensorRT conversion and Triton ``config.pbtxt`` generation."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from scrfd_constants import (
    FPN_STRIDES,
    INPUT_SIZE,
    MAX_BATCH_SIZE,
    NUM_ANCHORS,
    TRITON_MODEL_NAME,
    WORKSPACE_GB,
)


logger = logging.getLogger(__name__)


def convert_to_tensorrt(
    onnx_path: Path,
    plan_path: Path,
    fp16: bool = True,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> bool:
    """Convert SCRFD ONNX to TensorRT engine with dynamic batch."""
    logger.info(f'Converting to TensorRT: {onnx_path} -> {plan_path}')
    logger.info(f'  FP16: {fp16}, Max batch: {max_batch_size}')

    try:
        import tensorrt as trt
        from trt_utils import bake_fp16_onnx, create_explicit_network, enable_fp16

        trt.init_libnvinfer_plugins(None, '')
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network = create_explicit_network(builder)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        logger.info('  Baking FP16 (ModelOpt AutoCast, TRT 11 typed builds)...')
        onnx_path = bake_fp16_onnx(onnx_path)
        logger.info('  Parsing ONNX...')
        with Path(onnx_path).open('rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f'  Parse error {i}: {parser.get_error(i)}')
                raise RuntimeError('Failed to parse ONNX')

        logger.info(
            f'  Parsed: {network.num_layers} layers, '
            f'{network.num_inputs} inputs, {network.num_outputs} outputs'
        )

        # Configure
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)

        if fp16 and enable_fp16(builder, config):
            logger.info('  FP16 mode enabled')

        # Optimization profile for dynamic batch
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        opt_batch = max(1, max_batch_size // 4)
        profile.set_shape(
            input_name,
            min=(1, 3, INPUT_SIZE, INPUT_SIZE),
            opt=(opt_batch, 3, INPUT_SIZE, INPUT_SIZE),
            max=(max_batch_size, 3, INPUT_SIZE, INPUT_SIZE),
        )
        config.add_optimization_profile(profile)

        logger.info(f'  Profile: min=1, opt={opt_batch}, max={max_batch_size}')

        # Build
        logger.info('  Building TensorRT engine (may take 5-15 minutes)...')
        start_time = time.time()
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError('Failed to build TensorRT engine')

        build_time = time.time() - start_time
        logger.info(f'  Build time: {build_time:.1f}s')

        # Save
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(plan_path).open('wb') as f:
            f.write(serialized_engine)

        size_mb = plan_path.stat().st_size / (1024 * 1024)
        logger.info(f'  TensorRT saved: {plan_path} ({size_mb:.1f} MB)')

        return True

    except ImportError:
        logger.error('TensorRT not available. Run inside api container.')
        return False
    except Exception as e:
        logger.error(f'TensorRT conversion failed: {e}')
        import traceback

        traceback.print_exc()
        return False


def create_triton_config(plan_path: Path, max_batch_size: int = MAX_BATCH_SIZE) -> Path:
    """Create Triton config.pbtxt for SCRFD model.

    The ONNX is batch-patched (make_batch_dynamic) so the TensorRT engine
    supports dynamic batch. We use Triton's implicit batching (max_batch_size > 0)
    with dims that exclude the batch dimension. Triton manages the batch dimension.

    Anchor counts per stride:
      stride 8:  (640/8)^2  * 2 = 12800
      stride 16: (640/16)^2 * 2 = 3200
      stride 32: (640/32)^2 * 2 = 800
    """
    config_path = plan_path.parent.parent / 'config.pbtxt'

    # Compute anchor counts for fixed output dims
    fm_sizes = {s: INPUT_SIZE // s for s in FPN_STRIDES}
    n_anchors = {s: fm_sizes[s] * fm_sizes[s] * NUM_ANCHORS for s in FPN_STRIDES}

    config_content = f"""# SCRFD-10G Face Detection with 5-point Landmarks
# Input: RGB images, normalized (x-127.5)/128.0
# Output: 9 tensors across 3 FPN strides for CPU post-processing
# Post-processing: anchor decode + NMS in numpy (src/utils/scrfd_decode.py)
#
# The ONNX is batch-patched (export_scrfd.py) so the TensorRT engine supports
# dynamic batch via Triton's implicit batching (max_batch_size > 0).
# Anchor counts per stride: {n_anchors[8]} (stride 8), {n_anchors[16]} (stride 16), {n_anchors[32]} (stride 32).

name: "{TRITON_MODEL_NAME}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input.1"
    data_type: TYPE_FP32
    dims: [ 3, {INPUT_SIZE}, {INPUT_SIZE} ]
  }}
]

output [
  {{
    name: "score_8"
    data_type: TYPE_FP32
    dims: [ {n_anchors[8]}, 1 ]
  }},
  {{
    name: "score_16"
    data_type: TYPE_FP32
    dims: [ {n_anchors[16]}, 1 ]
  }},
  {{
    name: "score_32"
    data_type: TYPE_FP32
    dims: [ {n_anchors[32]}, 1 ]
  }},
  {{
    name: "bbox_8"
    data_type: TYPE_FP32
    dims: [ {n_anchors[8]}, 4 ]
  }},
  {{
    name: "bbox_16"
    data_type: TYPE_FP32
    dims: [ {n_anchors[16]}, 4 ]
  }},
  {{
    name: "bbox_32"
    data_type: TYPE_FP32
    dims: [ {n_anchors[32]}, 4 ]
  }},
  {{
    name: "kps_8"
    data_type: TYPE_FP32
    dims: [ {n_anchors[8]}, 10 ]
  }},
  {{
    name: "kps_16"
    data_type: TYPE_FP32
    dims: [ {n_anchors[16]}, 10 ]
  }},
  {{
    name: "kps_32"
    data_type: TYPE_FP32
    dims: [ {n_anchors[32]}, 10 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    # Count 1 is the default core loadout's baseline (~1.7 GB);
    # raise on a card with headroom to spare (see README "GPU sizing").
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

version_policy {{
  latest {{
    num_versions: 1
  }}
}}
"""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    logger.info(f'Triton config written: {config_path}')

    return config_path

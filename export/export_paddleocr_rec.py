#!/usr/bin/env python3
"""
PP-OCRv5 Multilingual Recognition Model - Verify & Convert to TensorRT.

This script verifies and optionally converts the PP-OCRv5 multilingual recognition
model ONNX to TensorRT for high-performance text recognition on GPU.

Model: PP-OCRv5 Multilingual Recognition (SVTR-LCNet architecture)
- Input: [B, 3, 48, W] text crop images (W: 48-2048, dynamic)
- Output: [B, T, 18385] character sequence logits (T dynamic, 18385 = 18383 chars + blank + space)
- Preprocessing: (x / 127.5) - 1, BGR format

The multilingual ONNX is downloaded by download_paddleocr.py (no PaddleX needed).
This script verifies the ONNX and builds the TensorRT engine in-process with the
TensorRT Python API -- self-contained, no docker CLI / Triton container dependency
(see convert_to_tensorrt_via_python_api). scripts/export_paddleocr.sh is the
alternative HOST-side path (docker compose run ... trtexec), for when you'd
rather build from outside any container.

Usage:
    python export/export_paddleocr_rec.py               # Verify + TRT convert
    python export/export_paddleocr_rec.py --skip-tensorrt  # Verify only
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# Model configuration
MODEL_NAME = 'PP-OCRv5_mobile_rec'
INPUT_NAME = 'x'
OUTPUT_NAME = 'fetch_name_0'

# Fixed height, dynamic width
REC_HEIGHT = 48
MIN_WIDTH = 48  # Minimum text crop width
OPT_WIDTH = 320  # Optimal width for TensorRT
MAX_WIDTH = 2048  # Maximum width for long text

# Dynamic output
NUM_CHARS = 18385  # 18383 multilingual chars + blank + space

# Batch sizes (recognition processes many text crops per image)
MIN_BATCH = 1
OPT_BATCH = 32  # Higher optimal batch for text crops
MAX_BATCH = 64  # Match Triton max_batch_size

# Paths (relative to project root, which is one level up from this script's directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
MODELS_DIR = _PROJECT_DIR / 'models'

ONNX_PATH = (
    Path('/app/pytorch_models/paddleocr/ppocr_rec_v5_mobile.onnx')
    if Path('/app').exists()
    else _PROJECT_DIR / 'pytorch_models/paddleocr/ppocr_rec_v5_mobile.onnx'
)
PLAN_OUTPUT = MODELS_DIR / 'paddleocr_rec_trt/1/model.plan'


def check_onnx_exists() -> bool:
    """Check if the multilingual ONNX model has been downloaded."""
    return ONNX_PATH.exists() and ONNX_PATH.stat().st_size > 0


def verify_onnx_model(onnx_path: Path) -> dict | None:
    """Verify ONNX model structure."""
    print('\n' + '=' * 60)
    print('Step 1: Verify ONNX Model')
    print('=' * 60)

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        input_info = model.graph.input[0]
        input_name = input_info.name
        input_shape = [
            d.dim_param if d.dim_param else d.dim_value
            for d in input_info.type.tensor_type.shape.dim
        ]

        output_info = model.graph.output[0]
        output_name = output_info.name
        output_shape = [
            d.dim_param if d.dim_param else d.dim_value
            for d in output_info.type.tensor_type.shape.dim
        ]

        print(f'  Input: {input_name} {input_shape}')
        print(f'  Output: {output_name} {output_shape}')
        print(f'  Nodes: {len(model.graph.node)}')

        # Verify dynamic shapes
        has_dynamic = any(isinstance(d, str) for d in input_shape)
        print(f'  Dynamic shapes: {has_dynamic}')

        return {
            'input_name': input_name,
            'input_shape': input_shape,
            'output_name': output_name,
            'output_shape': output_shape,
            'has_dynamic': has_dynamic,
        }

    except Exception as e:
        print(f'ERROR: ONNX verification failed: {e}')
        return None


def test_onnx_inference(onnx_path: Path) -> bool:
    """Test ONNX model with different widths."""
    print('\n' + '=' * 60)
    print('Step 2: Test ONNX Inference')
    print('=' * 60)

    try:
        import onnxruntime as ort

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        print(f'  Provider: {session.get_providers()[0]}')

        # Test with different widths (dynamic)
        test_shapes = [
            (1, 3, REC_HEIGHT, MIN_WIDTH),  # Min width
            (1, 3, REC_HEIGHT, OPT_WIDTH),  # Optimal width
            (8, 3, REC_HEIGHT, OPT_WIDTH),  # Optimal batch
            (1, 3, REC_HEIGHT, 640),  # Medium width
            (4, 3, REC_HEIGHT, 1280),  # Large width
        ]

        for shape in test_shapes:
            try:
                test_input = np.random.randn(*shape).astype(np.float32)
                output = session.run(None, {INPUT_NAME: test_input})[0]
                print(f'  {shape} -> {output.shape} OK')
            except Exception as e:
                print(f'  {shape} FAILED: {e}')
                return False

        return True

    except Exception as e:
        print(f'ERROR: ONNX test failed: {e}')
        return False


def unload_models_for_memory():
    """Unload Triton models to free GPU memory for TensorRT build."""
    import logging
    import time

    import requests

    logger = logging.getLogger(__name__)
    print('\nFreeing GPU memory by unloading models...')

    models_to_unload = [
        'ocr_pipeline',
        'paddleocr_det_trt',
        'paddleocr_rec_trt',
        'yolov11_small_trt_end2end',
        'scrfd_10g_bnkps',
        'arcface_w600k_r50',
        'mobileclip2_s2_image_encoder',
        'mobileclip2_s2_text_encoder',
    ]

    for model in models_to_unload:
        try:
            requests.post(f'http://localhost:4600/v2/repository/models/{model}/unload', timeout=5)
        except Exception as e:
            # Silently continue if unload fails - model may not be loaded
            logger.debug('Could not unload %s: %s', model, e)

    time.sleep(2)
    print('  Models unloaded')


def convert_to_tensorrt_via_python_api(onnx_path: Path, plan_path: Path) -> Path | None:
    """Build the TensorRT engine in-process with the TensorRT Python API.

    F-17 (fresh-start E2E findings 2026-09-25): this script runs inside
    ``yolo-api`` (see the module docstring's ``Usage``), which has no
    docker CLI or docker.sock, and shelling out to
    ``docker exec <TRITON_CONTAINER> trtexec ...`` also hardcoded the
    pre-G-01 container name ``triton-server`` -- after the project-name
    rename it's ``${COMPOSE_PROJECT_NAME}-triton``, so this always
    printed "triton-server container is not running" even with Triton
    up. yolo-api already depends on ``tensorrt-cu13`` directly (see
    requirements.txt; ``export/export_models.py`` builds YOLO engines
    the same way), so there's no need for Triton to be running, or even
    exist, at export time -- only the ONNX file and a GPU.
    """
    print('\n' + '=' * 60)
    print('Step 3: Convert to TensorRT')
    print('=' * 60)

    # Best-effort headroom for the build; a plain HTTP call, harmless
    # (and a no-op) if Triton isn't reachable. This only frees Triton's
    # in-memory model instances -- it never touches model.plan on disk,
    # so a build failure below still leaves whatever plan already exists.
    unload_models_for_memory()

    # F-17(b) (fresh-start E2E findings 2026-09-25, round 2): this used to
    # unlink the existing plan_path before attempting the build, so a
    # failed rebuild left the model with NO plan at all -- destroying a
    # working install. Build to a temp file, validate it, then swap it
    # into place atomically (see trt_utils.atomic_write_plan); the old
    # plan is only ever replaced by a plan that's confirmed to work.
    min_shape = (MIN_BATCH, 3, REC_HEIGHT, MIN_WIDTH)
    opt_shape = (OPT_BATCH, 3, REC_HEIGHT, OPT_WIDTH)
    max_shape = (MAX_BATCH, 3, REC_HEIGHT, MAX_WIDTH)
    print('Building with dynamic shapes:')
    print(f'  Min: {INPUT_NAME}:{min_shape}')
    print(f'  Opt: {INPUT_NAME}:{opt_shape}')
    print(f'  Max: {INPUT_NAME}:{max_shape}')
    print('  Memory Pool: 8192 MiB')
    print('\nThis may take 10-20 minutes for dynamic shapes...')

    try:
        import tensorrt as trt
        from trt_utils import atomic_write_plan, create_explicit_network

        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, '')
        builder = trt.Builder(trt_logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8192 * (1 << 20))

        # F-17(a) (fresh-start E2E findings 2026-09-25, round 2):
        # NetworkDefinitionCreationFlag.EXPLICIT_BATCH was removed in
        # TRT 11 (networks are always explicit-batch now); the hardcoded
        # flag lookup raised AttributeError on the image's TRT 11.1,
        # regressing the F-17 fix. create_explicit_network passes the
        # flag only where it still exists (TRT <= 10.x) -- same helper
        # export_models.py uses, which already works on TRT 11.1.
        network = create_explicit_network(builder)
        parser = trt.OnnxParser(network, trt_logger)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f'  ONNX parse error: {parser.get_error(i)}')
                return None

        profile = builder.create_optimization_profile()
        # NOTE (G-21): TRT 11.1 is strongly typed -- this ONNX is not
        # pre-baked to FP16, so the engine builds FP32 (TF32 tensor cores
        # on Ampere+); bake precision into the ONNX first for FP16.
        profile.set_shape(INPUT_NAME, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print('\nERROR: TensorRT engine build failed (see log above)')
            return None

        try:
            atomic_write_plan(bytes(serialized_engine), plan_path)
        except ValueError as e:
            print(f'\nERROR: built engine failed validation, keeping existing plan: {e}')
            return None

        print('\nTensorRT conversion successful!')
        print(f'  Engine size: {plan_path.stat().st_size / 1024 / 1024:.2f} MB')
        return plan_path

    except Exception as e:
        print(f'\nERROR: TensorRT conversion failed: {e}')
        import traceback

        traceback.print_exc()
        return None


def create_triton_config(plan_path: Path):
    """Create Triton config.pbtxt with dynamic dimensions."""
    print('\n' + '=' * 60)
    print('Step 4: Create Triton Config')
    print('=' * 60)

    config_path = plan_path.parent.parent / 'config.pbtxt'

    config_content = f"""# PP-OCRv5 Multilingual Text Recognition Model (TensorRT)
#
# SVTR-LCNet architecture for text sequence recognition
# Supports dynamic width for variable-length text recognition
# Multilingual model: Chinese + English + symbols (18385 character classes)
#
# Input:
#   - x: [B, 3, 48, W] FP32, preprocessed (x / 127.5 - 1), BGR
#        Text crops with height=48, width={MIN_WIDTH}-{MAX_WIDTH}
#
# Output:
#   - fetch_name_0: [B, T, {NUM_CHARS}] FP32, character probabilities
#        T timesteps (dynamic), {NUM_CHARS} character classes (multilingual dict)

name: "paddleocr_rec_trt"
platform: "tensorrt_plan"
max_batch_size: {MAX_BATCH}

input [
  {{
    name: "{INPUT_NAME}"
    data_type: TYPE_FP32
    dims: [ 3, {REC_HEIGHT}, -1 ]
  }}
]

output [
  {{
    name: "{OUTPUT_NAME}"
    data_type: TYPE_FP32
    dims: [ -1, {NUM_CHARS} ]
  }}
]

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }}
]

# Higher queue delay for recognition to accumulate text crops from multiple images
dynamic_batching {{
  preferred_batch_size: [ 16, 32, 64 ]
  max_queue_delay_microseconds: 10000
}}
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f'  Config: {config_path}')
    print(f'  Dynamic width: {MIN_WIDTH}-{MAX_WIDTH}')
    print(f'  Max batch: {MAX_BATCH}')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Verify & convert PP-OCRv5 multilingual recognition to TensorRT'
    )
    parser.add_argument(
        '--skip-tensorrt',
        action='store_true',
        help='Skip TensorRT conversion (verify ONNX only)',
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print('=' * 70)
    print('PP-OCRv5 Multilingual Recognition Model - Verify & Convert')
    print(f'  Model: {MODEL_NAME}')
    print(f'  Dynamic width: {MIN_WIDTH} - {MAX_WIDTH}')
    print(f'  Characters: {NUM_CHARS} (18383 multilingual + blank + space)')
    print('=' * 70)

    # Check ONNX exists (downloaded by download_paddleocr.py)
    if not check_onnx_exists():
        print(f'\nERROR: ONNX model not found: {ONNX_PATH}')
        print('\nDownload with: python export/download_paddleocr.py')
        return 1

    onnx_path = ONNX_PATH
    print(f'\nONNX model: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)')

    # Step 1: Verify ONNX
    model_info = verify_onnx_model(onnx_path)
    if not model_info:
        return 1

    # Step 2: Test ONNX
    if not test_onnx_inference(onnx_path):
        return 1

    # Step 3: Convert to TensorRT
    plan_path = None
    if not args.skip_tensorrt:
        plan_path = convert_to_tensorrt_via_python_api(onnx_path, PLAN_OUTPUT)
        if not plan_path:
            print('\nERROR: TensorRT conversion failed')
            print('  Check GPU memory and retry')
            return 1

    # Step 4: Create Triton config
    if plan_path:
        create_triton_config(plan_path)

    # Summary
    print('\n' + '=' * 70)
    print('Complete!')
    print('=' * 70)
    print('\nOutputs:')
    print(f'  ONNX:       {onnx_path}')
    if plan_path:
        print(f'  TensorRT:   {plan_path}')

    print('\nModel specifications:')
    print(f'  - Input: x [B, 3, 48, W] FP32 (W: {MIN_WIDTH}-{MAX_WIDTH})')
    print(f'  - Output: fetch_name_0 [B, T, {NUM_CHARS}] FP32')
    print('  - Preprocessing: (x / 127.5) - 1, BGR format')
    print(f'  - Max batch size: {MAX_BATCH}')

    if plan_path:
        print('\nNext steps:')
        print('  1. Reload Triton models:')
        print('     curl -X POST localhost:4600/v2/repository/models/paddleocr_rec_trt/load')
        print('  2. Test OCR: curl -X POST http://localhost:4603/ocr/predict -F "image=@test.png"')

    return 0


if __name__ == '__main__':
    sys.exit(main())

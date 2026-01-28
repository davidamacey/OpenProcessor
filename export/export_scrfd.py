#!/usr/bin/env python3
"""
Export SCRFD face detection model to TensorRT for Triton deployment.

SCRFD (Sample and Computation Redistribution for Face Detection) from InsightFace
provides face detection with 5-point landmarks, enabling proper Umeyama alignment
for ArcFace recognition.

Model: SCRFD-10G with Batch Normalization and Keypoints (scrfd_10g_bnkps)
- Input: [B, 3, 640, 640] FP32, RGB, normalized (x-127.5)/128
- Output: 9 tensors across 3 FPN strides (8, 16, 32):
  - score_{8,16,32}: face confidence per anchor
  - bbox_{8,16,32}: distance predictions (left, top, right, bottom)
  - kps_{8,16,32}: landmark offsets (5 points x 2 coords)
- WiderFace: 95.2% Easy / 93.9% Medium / 83.1% Hard

Key steps:
1. Download ONNX model from HuggingFace
2. Fix batch dimension for dynamic batching
3. Validate with ONNX Runtime
4. Convert to TensorRT FP16
5. Create Triton configuration

Usage:
    # From yolo-api container:
    docker compose exec yolo-api python /app/export/export_scrfd.py

    # From host with venv:
    python export/export_scrfd.py

    # Skip download if ONNX exists:
    python export/export_scrfd.py --skip-download

    # ONNX-only (no TensorRT):
    python export/export_scrfd.py --skip-trt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

INPUT_SIZE = 640
MAX_BATCH_SIZE = 64  # Must match max profile's max_batch_size
FP16_MODE = True
WORKSPACE_GB = 4

# Paths
ONNX_DIR = Path('/app/pytorch_models') if Path('/app').exists() else Path('pytorch_models')
ONNX_FILENAME = 'scrfd_10g_bnkps.onnx'
TRITON_MODEL_NAME = 'scrfd_10g_bnkps'
OUTPUT_DIR = Path('/app/models') if Path('/app').exists() else Path('models')

# Download URLs (in priority order)
DOWNLOAD_URLS = [
    'https://huggingface.co/LPDoctor/insightface/resolve/main/scrfd_10g_bnkps.onnx',
    'https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx',
]

# FPN configuration for SCRFD-10G
FPN_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2

# Expected output tensor names
SCORE_NAMES = [f'score_{s}' for s in FPN_STRIDES]
BBOX_NAMES = [f'bbox_{s}' for s in FPN_STRIDES]
KPS_NAMES = [f'kps_{s}' for s in FPN_STRIDES]
ALL_OUTPUT_NAMES = SCORE_NAMES + BBOX_NAMES + KPS_NAMES


# =============================================================================
# Download
# =============================================================================


def download_onnx(output_path: Path, force: bool = False) -> bool:
    """Download SCRFD ONNX model from HuggingFace."""
    from urllib.error import HTTPError, URLError
    from urllib.request import urlretrieve

    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        if 15 < size_mb < 20:  # Expected ~17MB
            logger.info(f'ONNX already exists: {output_path} ({size_mb:.1f} MB)')
            return True
        logger.info(f'Existing file wrong size ({size_mb:.1f} MB), re-downloading')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(DOWNLOAD_URLS):
        logger.info(f'Downloading from source {i + 1}/{len(DOWNLOAD_URLS)}...')
        logger.info(f'  URL: {url[:80]}...')
        try:
            start = time.time()
            urlretrieve(url, output_path)
            elapsed = time.time() - start
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f'  Downloaded: {size_mb:.1f} MB in {elapsed:.1f}s')
            return True
        except (URLError, HTTPError) as e:
            logger.warning(f'  Failed: {e}')
            continue

    logger.error('Failed to download from all sources')
    return False


# =============================================================================
# ONNX Analysis and Batch Fix
# =============================================================================


def analyze_onnx(onnx_path: Path) -> dict:
    """Analyze SCRFD ONNX model and return structure info."""
    import onnx

    logger.info(f'Analyzing ONNX: {onnx_path}')
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    logger.info(f'  Nodes: {len(model.graph.node)}, Opset: {model.opset_import[0].version}')

    info = {'inputs': {}, 'outputs': {}}

    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            elif d.dim_param:
                shape.append(d.dim_param)
            else:
                shape.append(-1)
        info['inputs'][inp.name] = shape
        logger.info(f'  Input: {inp.name} {shape}')

    for out in model.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            elif d.dim_param:
                shape.append(d.dim_param)
            else:
                shape.append(-1)
        info['outputs'][out.name] = shape
        logger.info(f'  Output: {out.name} {shape}')

    # Verify expected outputs
    output_names = set(info['outputs'].keys())
    expected = set(ALL_OUTPUT_NAMES)
    if expected.issubset(output_names):
        logger.info('  All 9 expected output tensors found')
    else:
        missing = expected - output_names
        logger.warning(f'  Missing outputs: {missing}')
        logger.info(f'  Available outputs: {output_names}')

    return info


def make_batch_dynamic(onnx_path: Path, output_path: Path | None = None) -> Path:
    """
    Fix SCRFD ONNX to support dynamic batch dimension with Triton batching.

    Standard InsightFace SCRFD ONNX has two issues preventing batch support:
    1. Transpose uses perm=[2,3,0,1] (puts spatial dims first, batch gets absorbed)
    2. Reshape uses [-1, D] (flattens batch into first dim)

    This function fixes both:
    1. Renames numeric output tensors to semantic names (score_8, bbox_16, etc.)
    2. Changes Transpose perm to [0,2,3,1] (keeps batch first: [B,H,W,C])
    3. Changes Reshape from [-1,D] to [0,-1,D] (preserves batch: [B,N_anchors,D])
    4. Sets dynamic batch dimension on input and output tensor specs

    Result: outputs go from [N_anchors, D] to [batch, N_anchors, D], enabling
    Triton's dynamic_batching scheduler.

    Args:
        onnx_path: Input ONNX path
        output_path: Output path (default: adds _batched suffix)

    Returns:
        Path to batch-compatible ONNX model
    """
    import onnx
    from onnx import TensorProto, helper

    logger.info('Patching ONNX for batch support (rename + transpose + reshape)...')

    model = onnx.load(str(onnx_path))

    # --- Step 1: Rename outputs from numeric to semantic names ---
    score_outputs = []
    bbox_outputs = []
    kps_outputs = []

    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        last_dim = dims[-1] if dims else 0
        if last_dim == 1:
            score_outputs.append(out)
        elif last_dim == 4:
            bbox_outputs.append(out)
        elif last_dim == 10:
            kps_outputs.append(out)

    rename_map = {}
    for outputs, prefix in [(score_outputs, 'score'), (bbox_outputs, 'bbox'), (kps_outputs, 'kps')]:
        outputs.sort(key=lambda o: o.type.tensor_type.shape.dim[0].dim_value, reverse=True)
        for out, stride in zip(outputs, FPN_STRIDES, strict=True):
            new_name = f'{prefix}_{stride}'
            if out.name != new_name:
                rename_map[out.name] = new_name
                logger.info(f'  Rename: {out.name} -> {new_name}')

    for node in model.graph.node:
        for i, output_name in enumerate(node.output):
            if output_name in rename_map:
                node.output[i] = rename_map[output_name]

    # --- Step 2: Build producer map and find output-connected ops ---
    producer = {}
    for node in model.graph.node:
        for o in node.output:
            producer[o] = node

    output_names = set()
    for out in model.graph.output:
        name = rename_map.get(out.name, out.name)
        output_names.add(name)

    # Find Transpose nodes that feed into output-connected Reshape nodes
    transpose_ids = set()
    reshape_init_names = set()
    for out_name in output_names:
        node = producer.get(out_name)
        if not node:
            continue
        current = node
        for _ in range(3):
            if current.op_type == 'Reshape':
                reshape_init_names.add(current.input[1])
            if current.op_type == 'Transpose':
                transpose_ids.add(id(current))
                break
            if current.input:
                prev = producer.get(current.input[0])
                if prev:
                    current = prev
                else:
                    break
            else:
                break

    # --- Step 3: Fix Transpose perm [2,3,0,1] -> [0,2,3,1] ---
    fixed_transposes = 0
    for node in model.graph.node:
        if node.op_type == 'Transpose' and id(node) in transpose_ids:
            for attr in node.attribute:
                if attr.name == 'perm' and list(attr.ints) == [2, 3, 0, 1]:
                    attr.ints[:] = [0, 2, 3, 1]
                    fixed_transposes += 1
    logger.info(f'  Fixed {fixed_transposes} Transpose nodes: perm [2,3,0,1] -> [0,2,3,1]')

    # --- Step 4: Fix Reshape [-1, D] -> [0, -1, D] ---
    fixed_reshapes = 0
    for init in model.graph.initializer:
        if init.name in reshape_init_names:
            import numpy as np

            old_shape = np.frombuffer(init.raw_data, dtype=np.int64)
            if len(old_shape) == 2 and old_shape[0] == -1:
                new_shape = np.array([0, -1, old_shape[1]], dtype=np.int64)
                init.raw_data = new_shape.tobytes()
                init.dims[:] = [len(new_shape)]
                fixed_reshapes += 1
                logger.info(f'  Reshape {init.name}: {old_shape} -> {new_shape}')
    logger.info(f'  Fixed {fixed_reshapes} Reshape initializers')

    # --- Step 5: Update input with dynamic batch ---
    input_tensor = model.graph.input[0]
    new_input = helper.make_tensor_value_info(
        input_tensor.name, TensorProto.FLOAT, ['batch', 3, INPUT_SIZE, INPUT_SIZE]
    )
    model.graph.input.remove(input_tensor)
    model.graph.input.insert(0, new_input)

    # --- Step 6: Update outputs with [batch, N_anchors, D] shapes ---
    fm_sizes = {s: INPUT_SIZE // s for s in FPN_STRIDES}
    n_anchors_map = {s: fm_sizes[s] * fm_sizes[s] * NUM_ANCHORS for s in FPN_STRIDES}
    last_dim_map = {'score': 1, 'bbox': 4, 'kps': 10}

    new_outputs = []
    for out in model.graph.output:
        name = rename_map.get(out.name, out.name)
        prefix = name.split('_')[0]
        stride = int(name.split('_')[1])
        n_anchors = n_anchors_map[stride]
        last_dim = last_dim_map[prefix]
        new_out = helper.make_tensor_value_info(
            name, TensorProto.FLOAT, ['batch', n_anchors, last_dim]
        )
        new_outputs.append(new_out)

    while len(model.graph.output) > 0:
        model.graph.output.pop()
    for new_out in new_outputs:
        model.graph.output.append(new_out)

    # --- Save ---
    if output_path is None:
        output_path = onnx_path.parent / f'{onnx_path.stem}_batched.onnx'

    onnx.save(model, str(output_path))
    logger.info(f'  Batch-compatible model saved: {output_path}')

    return output_path


# =============================================================================
# ONNX Runtime Validation
# =============================================================================


def test_onnx_inference(onnx_path: Path, batch_size: int = 1) -> bool:
    """Test SCRFD inference with ONNX Runtime."""
    logger.info(f'Testing ONNX inference (batch={batch_size})...')

    try:
        import onnxruntime as ort

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        active_provider = session.get_providers()[0]
        logger.info(f'  Provider: {active_provider}')

        # Input info
        input_info = session.get_inputs()[0]
        logger.info(f'  Input: {input_info.name} {input_info.shape}')

        # Create test input (normalized)
        test_input = np.random.randint(0, 256, (batch_size, 3, INPUT_SIZE, INPUT_SIZE))
        test_input = (test_input.astype(np.float32) - 127.5) / 128.0

        # Run inference
        start = time.time()
        outputs = session.run(None, {input_info.name: test_input})
        inference_time = (time.time() - start) * 1000

        logger.info(f'  Inference time: {inference_time:.2f}ms (batch={batch_size})')

        # Log output shapes
        output_names = [o.name for o in session.get_outputs()]
        for name, arr in zip(output_names, outputs, strict=False):
            logger.info(f'  Output {name}: shape={arr.shape}, dtype={arr.dtype}')

        # Verify 9 outputs for KPS model
        if len(outputs) == 9:
            logger.info('  9 output tensors confirmed (scores + boxes + landmarks)')
        elif len(outputs) == 6:
            logger.warning('  Only 6 outputs (no landmarks) - wrong model variant?')
        else:
            logger.warning(f'  Unexpected {len(outputs)} outputs')

        return True

    except Exception as e:
        logger.error(f'ONNX inference test failed: {e}')
        return False


def benchmark_onnx(onnx_path: Path, iterations: int = 100) -> None:
    """Benchmark SCRFD ONNX inference at various batch sizes."""
    import onnxruntime as ort

    logger.info(f'Benchmarking ONNX ({iterations} iterations per batch size)...')

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    for batch_size in [1, 4, 8, 16, 32]:
        test_input = np.random.rand(batch_size, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        test_input = (test_input - 0.5) * 2  # rough normalize

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: test_input})

        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.time()
            session.run(None, {input_name: test_input})
            latencies.append((time.time() - start) * 1000)

        mean_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        per_img = mean_lat / batch_size
        fps = 1000 * batch_size / mean_lat

        logger.info(
            f'  Batch {batch_size:2d}: {mean_lat:.2f}ms total, '
            f'{per_img:.2f}ms/img, {fps:.0f} FPS (p95: {p95_lat:.2f}ms)'
        )


# =============================================================================
# TensorRT Conversion
# =============================================================================


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

        trt.init_libnvinfer_plugins(None, '')
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        logger.info('  Parsing ONNX...')
        with open(onnx_path, 'rb') as f:
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

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
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
        with open(plan_path, 'wb') as f:
            f.write(serialized_engine)

        size_mb = plan_path.stat().st_size / (1024 * 1024)
        logger.info(f'  TensorRT saved: {plan_path} ({size_mb:.1f} MB)')

        return True

    except ImportError:
        logger.error('TensorRT not available. Run inside yolo-api container.')
        return False
    except Exception as e:
        logger.error(f'TensorRT conversion failed: {e}')
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Triton Configuration
# =============================================================================


def create_triton_config(plan_path: Path, max_batch_size: int = MAX_BATCH_SIZE) -> Path:
    """Create Triton config.pbtxt for SCRFD model.

    Note: The standard InsightFace SCRFD ONNX has reshape ops that absorb the
    batch dimension into spatial dims (outputs are 2D: [-1, D]). This means
    max_batch_size must be 0 (explicit batch mode). Throughput is achieved via
    multiple GPU instances + high client concurrency, matching InsightFace-REST.
    """
    config_path = plan_path.parent.parent / 'config.pbtxt'

    config_content = f"""# SCRFD-10G Face Detection with 5-point Landmarks
# Input: RGB images, normalized (x-127.5)/128.0
# Output: 9 tensors across 3 FPN strides for CPU post-processing
# Post-processing: anchor decode + NMS in numpy (src/utils/scrfd_decode.py)
#
# Note: max_batch_size=0 (explicit batch mode) because the SCRFD ONNX model
# has reshape ops that absorb the batch dimension into spatial dims. Throughput
# is achieved via multiple GPU instances + high client concurrency, matching
# the InsightFace-REST architecture (~820 FPS on RTX 4090).

name: "{TRITON_MODEL_NAME}"
platform: "tensorrt_plan"
max_batch_size: 0

input [
  {{
    name: "input.1"
    data_type: TYPE_FP32
    dims: [ 1, 3, {INPUT_SIZE}, {INPUT_SIZE} ]
  }}
]

output [
  {{
    name: "score_8"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
  }},
  {{
    name: "score_16"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
  }},
  {{
    name: "score_32"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
  }},
  {{
    name: "bbox_8"
    data_type: TYPE_FP32
    dims: [ -1, 4 ]
  }},
  {{
    name: "bbox_16"
    data_type: TYPE_FP32
    dims: [ -1, 4 ]
  }},
  {{
    name: "bbox_32"
    data_type: TYPE_FP32
    dims: [ -1, 4 ]
  }},
  {{
    name: "kps_8"
    data_type: TYPE_FP32
    dims: [ -1, 10 ]
  }},
  {{
    name: "kps_16"
    data_type: TYPE_FP32
    dims: [ -1, 10 ]
  }},
  {{
    name: "kps_32"
    data_type: TYPE_FP32
    dims: [ -1, 10 ]
  }}
]

instance_group [
  {{
    count: 4
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


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Export SCRFD face detection to TensorRT for Triton',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + export):
  python export/export_scrfd.py

  # Skip download (ONNX already exists):
  python export/export_scrfd.py --skip-download

  # ONNX validation only (no TensorRT):
  python export/export_scrfd.py --skip-trt

  # With benchmark:
  python export/export_scrfd.py --benchmark
""",
    )
    parser.add_argument('--skip-download', action='store_true', help='Skip ONNX download')
    parser.add_argument('--skip-trt', action='store_true', help='Skip TensorRT conversion')
    parser.add_argument('--force-download', action='store_true', help='Re-download ONNX')
    parser.add_argument('--benchmark', action='store_true', help='Run ONNX benchmark')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16')
    parser.add_argument('--max-batch', type=int, default=MAX_BATCH_SIZE, help='Max batch size')
    args = parser.parse_args()

    logger.info('=' * 60)
    logger.info('SCRFD Face Detection Export')
    logger.info('=' * 60)

    onnx_path = ONNX_DIR / ONNX_FILENAME
    plan_dir = OUTPUT_DIR / TRITON_MODEL_NAME / '1'
    plan_path = plan_dir / 'model.plan'

    # Step 1: Download ONNX
    if not args.skip_download and not download_onnx(onnx_path, force=args.force_download):
        return 1

    if not onnx_path.exists():
        logger.error(f'ONNX not found: {onnx_path}')
        logger.info('Run: python export/download_face_models.py --models scrfd_10g_bnkps')
        return 1

    # Step 2: Analyze
    analyze_onnx(onnx_path)

    # Step 3: Make batch dynamic
    dynamic_onnx = make_batch_dynamic(onnx_path)

    # Step 4: Test inference
    test_onnx_inference(dynamic_onnx, batch_size=1)

    # Step 5: Benchmark
    if args.benchmark:
        benchmark_onnx(dynamic_onnx)

    # Step 6: Convert to TensorRT
    if not args.skip_trt:
        success = convert_to_tensorrt(
            dynamic_onnx,
            plan_path,
            fp16=not args.no_fp16,
            max_batch_size=args.max_batch,
        )
        if not success:
            return 1

        # Step 7: Create Triton config
        create_triton_config(plan_path, max_batch_size=args.max_batch)

    logger.info('')
    logger.info('=' * 60)
    logger.info('SCRFD Export Complete!')
    logger.info('=' * 60)
    logger.info(f'  ONNX: {onnx_path}')
    if not args.skip_trt:
        logger.info(f'  TensorRT: {plan_path}')
        logger.info(f'  Triton config: {plan_path.parent.parent / "config.pbtxt"}')
    logger.info('')
    logger.info('Next steps:')
    logger.info('  1. Restart Triton: docker compose restart triton-server')
    logger.info('  2. Load model: make load-scrfd')
    logger.info('  3. Test: python tests/test_scrfd_pipeline.py')

    return 0


if __name__ == '__main__':
    sys.exit(main())

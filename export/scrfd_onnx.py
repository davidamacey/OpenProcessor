"""SCRFD ONNX steps: download, structure analysis, batch-dimension fix, validation and benchmark."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scrfd_constants import ALL_OUTPUT_NAMES, DOWNLOAD_URLS, FPN_STRIDES, INPUT_SIZE, NUM_ANCHORS


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


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
            urlretrieve(url, output_path)  # nosec B310  URL comes from the hardcoded DOWNLOAD_URLS list
            elapsed = time.time() - start
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f'  Downloaded: {size_mb:.1f} MB in {elapsed:.1f}s')
            return True
        except (URLError, HTTPError) as e:
            logger.warning(f'  Failed: {e}')
            continue

    logger.error('Failed to download from all sources')
    return False


def analyze_onnx(onnx_path: Path) -> dict:
    """Analyze SCRFD ONNX model and return structure info."""
    import onnx

    logger.info(f'Analyzing ONNX: {onnx_path}')
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    logger.info(f'  Nodes: {len(model.graph.node)}, Opset: {model.opset_import[0].version}')

    info: dict[str, dict[str, Any]] = {'inputs': {}, 'outputs': {}}

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


def check_onnx_inference(onnx_path: Path, batch_size: int = 1) -> bool:
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

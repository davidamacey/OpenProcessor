#!/usr/bin/env python3
"""
YOLO26 Native Export Script
===========================

Export YOLO26 models for Triton deployment. YOLO26 is end-to-end NMS-free:
the fused export emits a single ``(batch, 300, 6)`` output tensor of
``[x1, y1, x2, y2, score, class]`` rows — no EfficientNMS plugin and no
ultralytics patch required (contrast with export_models.py, the legacy
YOLO11 EfficientNMS toolchain).

Requires ultralytics >= 8.4 (the main container environment).

Formats:
--------
1. onnx - Fused NMS-free ONNX (intermediate; single output tensor)
2. trt  - TensorRT engine built from that ONNX (deployed to Triton)

Usage:
------
# Export nano + small to TensorRT (default)
docker compose exec yolo-api python /app/export/export_yolo26.py

# Specific sizes / formats
docker compose exec yolo-api python /app/export/export_yolo26.py --models small --formats trt
"""

import argparse
import gc
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import onnx
import tensorrt as trt
import torch
from trt_utils import create_explicit_network
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Default configurations for YOLO26 model sizes. Triton names follow the
# yolo26_{size}_trt convention (single fused output — the API's adapter
# registry detects the format from Triton model metadata, not the name).
DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    'nano': {
        'pt_file': '/app/pytorch_models/yolo26n.pt',
        'triton_name': 'yolo26_nano',
        'max_batch': 128,
    },
    'small': {
        'pt_file': '/app/pytorch_models/yolo26s.pt',
        'triton_name': 'yolo26_small',
        'max_batch': 64,
    },
    'medium': {
        'pt_file': '/app/pytorch_models/yolo26m.pt',
        'triton_name': 'yolo26_medium',
        'max_batch': 32,
    },
    'large': {
        'pt_file': '/app/pytorch_models/yolo26l.pt',
        'triton_name': 'yolo26_large',
        'max_batch': 16,
    },
    'xlarge': {
        'pt_file': '/app/pytorch_models/yolo26x.pt',
        'triton_name': 'yolo26_xlarge',
        'max_batch': 8,
    },
}

IMG_SIZE = 640
WORKSPACE_GB = 4
MODELS_DIR = Path('/app/models')
EXPORT_DIR = Path('/app/pytorch_models')


# ============================================================================
# Helpers
# ============================================================================


def resolve_model(pt_file: str) -> YOLO:
    """Load the .pt, auto-downloading through ultralytics when missing.

    When the configured path is absent, ultralytics downloads the bare
    weight name into the current directory; the file is then moved to the
    configured location so subsequent runs are offline.
    """
    path = Path(pt_file)
    if path.exists():
        return YOLO(str(path))

    logger.info(f'{pt_file} not found — downloading {path.name} via ultralytics')
    model = YOLO(path.name)
    downloaded = Path(path.name)
    if downloaded.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(downloaded, path)
        return YOLO(str(path))
    return model


def export_fused_onnx(model: YOLO, config: dict[str, Any]) -> Path:
    """Export the fused NMS-free ONNX (single output tensor)."""
    logger.info(f'Exporting fused ONNX (imgsz={IMG_SIZE}, dynamic batch)...')
    exported = model.export(
        format='onnx',
        imgsz=IMG_SIZE,
        dynamic=True,
        batch=config['max_batch'],
        device='cpu',
        simplify=True,
    )
    onnx_path = Path(exported)

    target = EXPORT_DIR / f'{config["triton_name"]}.onnx'
    if onnx_path != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(onnx_path, target)
    logger.info(f'ONNX saved: {target}')
    return target


def inspect_fused_output(onnx_path: Path) -> tuple[str, list[int]]:
    """Return the fused output tensor's (name, per-item dims).

    YOLO26 fused export emits exactly one output shaped
    ``(batch, max_det, 6)``; the per-item dims (without the batch axis)
    feed straight into config.pbtxt.
    """
    graph = onnx.load(str(onnx_path)).graph
    outputs = list(graph.output)
    if len(outputs) != 1:
        raise RuntimeError(
            f'Expected a single fused output, got {[o.name for o in outputs]} — '
            'is this really a YOLO26 (end2end) export?'
        )
    out = outputs[0]
    dims = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim[1:]]
    logger.info(f'Fused output: name={out.name} per-item dims={dims}')
    return out.name, dims


def build_engine(onnx_path: Path, plan_path: Path, max_batch: int) -> bool:
    """Build a TensorRT engine with a dynamic-batch profile."""
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)
    network = create_explicit_network(builder)
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            logger.error(f'  ONNX parse error [{i}]: {parser.get_error(i)}')
        return False

    profile = builder.create_optimization_profile()
    min_shape = (1, 3, IMG_SIZE, IMG_SIZE)
    opt_shape = (max(1, max_batch // 2), 3, IMG_SIZE, IMG_SIZE)
    max_shape = (max_batch, 3, IMG_SIZE, IMG_SIZE)
    for i in range(network.num_inputs):
        profile.set_shape(network.get_input(i).name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info('FP16 precision enabled')

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info('Building TensorRT engine (this may take several minutes)...')
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        logger.error('Engine build failed - builder returned None')
        return False

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, 'wb') as f:
        f.write(serialized)
    logger.info(f'Engine saved: {plan_path} ({plan_path.stat().st_size / 1e6:.1f} MB)')
    return True


def write_triton_config(
    model_dir: Path,
    triton_name: str,
    max_batch: int,
    output_name: str,
    output_dims: list[int],
) -> None:
    """Write config.pbtxt for a fused single-output TensorRT model."""
    preferred = [size for size in [8, 16, 32, 64] if size <= max_batch] or [1]
    preferred_str = ', '.join(str(b) for b in preferred)
    dims_str = ', '.join(str(d) for d in output_dims)

    config = f"""name: "{triton_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch}

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, {IMG_SIZE}, {IMG_SIZE} ]
  }}
]

# Fused NMS-free output: one row per detection, [x1, y1, x2, y2, score, class]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ {dims_str} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_str} ]
  max_queue_delay_microseconds: 25000
}}

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'config.pbtxt').write_text(config)
    logger.info(f'Generated Triton config: {model_dir / "config.pbtxt"}')


def write_labels(model: YOLO, model_dir: Path) -> None:
    """Write labels.txt from the model's class-name map."""
    names = getattr(model, 'names', None)
    if not names:
        return
    ordered = [names[i] for i in sorted(names)]
    (model_dir / 'labels.txt').write_text('\n'.join(ordered) + '\n')


# ============================================================================
# Main
# ============================================================================


def export_model(size: str, config: dict[str, Any], formats: list[str]) -> dict[str, Any]:
    """Run the export pipeline for one model size."""
    result: dict[str, Any] = {'model': size, 'triton_name': config['triton_name']}
    logger.info('=' * 70)
    logger.info(f'Exporting YOLO26 {size} -> {config["triton_name"]}')
    logger.info('=' * 70)

    model = resolve_model(config['pt_file'])
    onnx_path = export_fused_onnx(model, config)
    output_name, output_dims = inspect_fused_output(onnx_path)
    result['onnx'] = str(onnx_path)

    if 'trt' in formats:
        triton_model = f'{config["triton_name"]}_trt'
        model_dir = MODELS_DIR / triton_model
        plan_path = model_dir / '1' / 'model.plan'
        if not build_engine(onnx_path, plan_path, config['max_batch']):
            result['status'] = 'failed'
            return result
        write_triton_config(model_dir, triton_model, config['max_batch'], output_name, output_dims)
        write_labels(model, model_dir)
        result['trt'] = str(plan_path)

    result['status'] = 'success'
    return result


def parse_custom_model(arg: str) -> tuple[str, dict[str, Any]]:
    """Parse a ``pt_path:triton_name[:max_batch]`` custom-model spec."""
    parts = arg.split(':')
    if len(parts) < 2:
        raise ValueError(f'--custom-model expects pt_path:triton_name[:max_batch], got {arg!r}')
    pt_file, triton_name = parts[0], parts[1]
    max_batch = int(parts[2]) if len(parts) > 2 else 64
    return triton_name, {'pt_file': pt_file, 'triton_name': triton_name, 'max_batch': max_batch}


def main() -> None:
    parser = argparse.ArgumentParser(description='Export YOLO26 models for Triton (NMS-free)')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=sorted(DEFAULT_MODELS),
        default=['nano', 'small'],
        help='Model sizes to export (default: nano small)',
    )
    parser.add_argument(
        '--custom-model',
        help='Export a custom end2end model: pt_path:triton_name[:max_batch]',
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'trt', 'all'],
        default=['trt'],
        help='Export formats (default: trt; onnx is always produced as intermediate)',
    )
    args = parser.parse_args()

    formats = ['onnx', 'trt'] if 'all' in args.formats else args.formats

    if args.custom_model:
        name, config = parse_custom_model(args.custom_model)
        results = [export_model(name, config, formats)]
    else:
        results = [export_model(size, DEFAULT_MODELS[size], formats) for size in args.models]

    failed = [r for r in results if r.get('status') != 'success']
    for r in results:
        logger.info(f'{r["model"]}: {r.get("status")} ({r.get("trt", r.get("onnx", "-"))})')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()

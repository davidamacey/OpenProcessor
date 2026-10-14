#!/usr/bin/env python3
"""
Dual-Head Detector Export (detections + backbone feature map)
=============================================================

Re-export a YOLO-family detector so its ONNX graph exposes **two** named
outputs instead of one:

    output0    : the detector head's raw tensor (layout is the family's own —
                 YOLOv5 ``[B, N, 5 + nc]``, Ultralytics v8+ ``[B, 4 + nc, N]``)
    sppf_feat  : the backbone bottleneck feature map, ``[B, C, H, W]``

Why this exists
---------------
The curation subsystem stores a per-item backbone embedding
(``backbone_embedding`` on the items index, dimension
``CurationConfig.backbone_embedding_dim``) and consumes it in residual
clustering, the 2-D embedding visualization, item scores and the OCC
conflict handler. That embedding is produced by RoI-pooling a detector's
backbone feature map over each detection box
(:func:`src.services.detection.geometry.roi_pool_sppf`), which needs the
feature map on the wire — a stock detector export only emits the
detection tensor. This script is the missing producer side.

Nothing here is tied to a particular model, class count or deployment:
the checkpoint, Triton model name, network input size, tapped module and
both output names are CLI arguments, mirroring how model identity is
parameterized elsewhere in this repo (``DetectionProfile.detector_model``
/ ``input_size``, ``export_yolo26.py --custom-model``).

Loaders
-------
``--loader ultralytics``  Ultralytics checkpoints/YAMLs (YOLOv8/11/26, ...).
``--loader yolov5``       A YOLOv5 fork checkout (``models/yolo.py`` on
                          disk, passed via ``--yolov5-fork``), needed only to
                          unpickle that fork's checkpoints. The serving path
                          does not need it: client-side NMS in
                          :mod:`src.services.detection.ensemble_nms` is native.
``--loader auto``         (default) Ultralytics first, fork as fallback.

Usage
-----
    # ONNX only (no GPU needed)
    docker compose exec api python /app/export/export_detector_dual_head.py \\
        --weights /app/pytorch_models/my_detector.pt \\
        --triton-name my_detector_dual_head

    # ONNX + TensorRT engine + config.pbtxt written into the model repo
    docker compose exec api python /app/export/export_detector_dual_head.py \\
        --weights /app/pytorch_models/my_detector.pt \\
        --triton-name my_detector_dual_head \\
        --imgsz 1280 --max-batch 16 --formats onnx trt

    # A YOLOv5-fork checkpoint
    docker compose exec api python /app/export/export_detector_dual_head.py \\
        --weights /app/pytorch_models/legacy_v5.pt --loader yolov5 \\
        --yolov5-fork /app/external/yolov5 --imgsz 1280 \\
        --triton-name legacy_v5_dual_head

``export_detector_dual_head.sh`` is the trtexec equivalent of the ``trt``
format, for environments that have ``trtexec`` but not the TensorRT Python
bindings (e.g. inside the triton-server container).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dual_head_model import (
    DEFAULT_DETECT_OUTPUT,
    DEFAULT_FEATURE_MODULE,
    DEFAULT_FEATURE_OUTPUT,
    DEFAULT_IMG_SIZE,
    DEFAULT_OPSET,
    export_dual_head_onnx,
    load_detector,
)
from dual_head_triton import build_engine, render_labels, render_triton_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('export_detector_dual_head')


# ============================================================================
# Configuration
# ============================================================================


DEFAULT_MAX_BATCH = 8
DEFAULT_MODELS_DIR = Path('/app/models')
DEFAULT_ONNX_DIR = Path('/app/pytorch_models')
DEFAULT_YOLOV5_FORK = './external/yolov5'


# Triton model names are directory names in the model repository.
TRITON_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


@dataclass
class DualHeadExport:
    """Report for one dual-head export run."""

    weights: str
    triton_name: str
    loader: str
    onnx: str
    imgsz: int
    opset: int
    feature_layer: str
    feature_channels: int
    feature_spatial: tuple[int, int]
    detect_dims: list[int]
    round_trip: str = 'skipped'
    plan: str | None = None
    config: str | None = None
    labels: str | None = None
    class_names: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            'weights': self.weights,
            'triton_name': self.triton_name,
            'loader': self.loader,
            'onnx': self.onnx,
            'imgsz': self.imgsz,
            'opset': self.opset,
            'feature_layer': self.feature_layer,
            'feature_channels': self.feature_channels,
            'feature_spatial': list(self.feature_spatial),
            'detect_dims': self.detect_dims,
            'round_trip': self.round_trip,
            'plan': self.plan,
            'config': self.config,
            'labels': self.labels,
            'num_classes': len(self.class_names),
        }


# ============================================================================
# Argument validation (pure — unit-tested without a GPU)
# ============================================================================


def validate_args(args: argparse.Namespace) -> None:
    """Reject argument combinations that cannot produce a loadable model.

    Raises:
        ValueError: With a message naming the offending flag. Checked here
            rather than after a multi-minute checkpoint load so a typo
            fails in milliseconds.
    """
    if not TRITON_NAME_RE.match(args.triton_name):
        msg = (
            f'--triton-name {args.triton_name!r} is not a valid Triton model '
            'name (it becomes a directory in the model repository): use '
            '[A-Za-z0-9._-] and do not start with punctuation.'
        )
        raise ValueError(msg)
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        msg = f'--imgsz must be a positive multiple of 32, got {args.imgsz}'
        raise ValueError(msg)
    if args.max_batch < 1:
        msg = f'--max-batch must be >= 1, got {args.max_batch}'
        raise ValueError(msg)
    if args.opset < 12:
        msg = f'--opset must be >= 12 for a dual-output dynamic-batch export, got {args.opset}'
        raise ValueError(msg)
    if args.feature_index is not None and args.feature_index < 0:
        msg = f'--feature-index must be >= 0, got {args.feature_index}'
        raise ValueError(msg)
    if args.detect_output == args.feature_output:
        msg = f'--detect-output and --feature-output must differ (both are {args.detect_output!r})'
        raise ValueError(msg)
    if args.loader == 'yolov5' and not (Path(args.yolov5_fork) / 'models' / 'yolo.py').exists():
        msg = (
            f'--loader yolov5 needs a fork checkout; {args.yolov5_fork}/models/yolo.py '
            f'not found. Pass --yolov5-fork.'
        )
        raise ValueError(msg)


def run(args: argparse.Namespace) -> DualHeadExport:
    """Execute the export described by ``args`` (already validated)."""
    weights = args.weights.resolve()
    if not weights.exists():
        msg = f'Checkpoint not found: {weights}'
        raise FileNotFoundError(msg)

    onnx_path = args.onnx or (args.onnx_dir / f'{args.triton_name}.onnx')
    model, class_names, loader_used = load_detector(weights, args.loader, args.yolov5_fork)
    onnx_report = export_dual_head_onnx(
        model,
        onnx_path,
        imgsz=args.imgsz,
        opset=args.opset,
        feature_module=args.feature_module,
        feature_index=args.feature_index,
        detect_output=args.detect_output,
        feature_output=args.feature_output,
        validate=not args.no_validate,
    )

    result = DualHeadExport(
        weights=str(weights),
        triton_name=args.triton_name,
        loader=loader_used,
        onnx=str(onnx_path),
        imgsz=args.imgsz,
        opset=args.opset,
        feature_layer=onnx_report['feature_layer'],
        feature_channels=onnx_report['feature_channels'],
        feature_spatial=onnx_report['feature_spatial'],
        detect_dims=onnx_report['detect_dims'],
        round_trip=onnx_report['round_trip'],
        class_names=[class_names[i] for i in sorted(class_names)],
    )

    if 'trt' not in args.formats:
        return result

    model_dir = args.models_dir / args.triton_name
    plan_path = model_dir / '1' / 'model.plan'
    if not build_engine(onnx_path, plan_path, imgsz=args.imgsz, max_batch=args.max_batch):
        msg = f'TensorRT engine build failed for {args.triton_name}'
        raise RuntimeError(msg)
    result.plan = str(plan_path)

    config_path = model_dir / 'config.pbtxt'
    config_path.write_text(
        render_triton_config(
            triton_name=args.triton_name,
            imgsz=args.imgsz,
            max_batch=args.max_batch,
            detect_output=args.detect_output,
            detect_dims=result.detect_dims,
            feature_output=args.feature_output,
            feature_channels=result.feature_channels,
            feature_spatial=result.feature_spatial,
        )
    )
    result.config = str(config_path)
    logger.info('Wrote %s', config_path)

    labels = render_labels(class_names)
    if labels:
        labels_path = model_dir / 'labels.txt'
        labels_path.write_text(labels)
        result.labels = str(labels_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Export a YOLO-family detector with a backbone feature-map output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--weights', type=Path, required=True, help='Detector checkpoint (.pt).')
    parser.add_argument(
        '--triton-name',
        required=True,
        help='Triton model name (also the model-repository directory name).',
    )
    parser.add_argument(
        '--loader',
        choices=['auto', 'ultralytics', 'yolov5'],
        default='auto',
        help='Checkpoint loader (default: auto — ultralytics, then the YOLOv5 fork).',
    )
    parser.add_argument(
        '--yolov5-fork',
        type=Path,
        default=Path(DEFAULT_YOLOV5_FORK),
        help=f'YOLOv5 fork checkout (default: {DEFAULT_YOLOV5_FORK}).',
    )
    parser.add_argument(
        '--feature-module',
        default=DEFAULT_FEATURE_MODULE,
        help=f'Class name of the module to tap (default: {DEFAULT_FEATURE_MODULE}).',
    )
    parser.add_argument(
        '--feature-index',
        type=int,
        default=None,
        help='Tap this layer index instead of searching by class name.',
    )
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMG_SIZE, help='Network input size.')
    parser.add_argument('--opset', type=int, default=DEFAULT_OPSET, help='ONNX opset version.')
    parser.add_argument(
        '--max-batch', type=int, default=DEFAULT_MAX_BATCH, help='Engine/Triton max batch size.'
    )
    parser.add_argument(
        '--detect-output',
        default=DEFAULT_DETECT_OUTPUT,
        help=f'Name of the detection output tensor (default: {DEFAULT_DETECT_OUTPUT}).',
    )
    parser.add_argument(
        '--feature-output',
        default=DEFAULT_FEATURE_OUTPUT,
        help=f'Name of the feature-map output tensor (default: {DEFAULT_FEATURE_OUTPUT}).',
    )
    parser.add_argument('--onnx', type=Path, default=None, help='Explicit ONNX output path.')
    parser.add_argument(
        '--onnx-dir',
        type=Path,
        default=DEFAULT_ONNX_DIR,
        help=f'Directory for the intermediate ONNX (default: {DEFAULT_ONNX_DIR}).',
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f'Triton model repository (default: {DEFAULT_MODELS_DIR}).',
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'trt'],
        default=['onnx'],
        help='Artifacts to produce (default: onnx; onnx is always produced).',
    )
    parser.add_argument(
        '--no-validate', action='store_true', help='Skip the onnxruntime round-trip check.'
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    try:
        validate_args(args)
    except ValueError as exc:
        logger.error('%s', exc)
        return 2

    result = run(args)
    logger.info('Export complete')
    for key, value in result.as_dict().items():
        logger.info('  %s: %s', key, value)
    logger.info(
        'Declare both outputs in config.pbtxt and RoI-pool %s per detection '
        '(src.services.detection.geometry.roi_pool_sppf, target_dim = '
        'CurationConfig.backbone_embedding_dim).',
        args.feature_output,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())

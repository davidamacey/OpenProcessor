"""Detector bake-off CLI: score one detector backend on a frozen test split.

Domain-agnostic: point it at any YOLO test split and any of the wired
backends. What is measured -- target class, the context classes a crop-mode
cascade crops on, the Triton model, backend and metric defaults -- comes
from a :class:`~scripts.curation.bakeoff.profile.BakeoffProfile`
(``--profile``; the neutral ``generic`` profile when omitted). Any
explicit CLI flag overrides the profile's value.

Example:
    .venv/bin/python -m scripts.curation.bakeoff.run \
        --profile my_profile.json \
        --dataset ./data/bakeoff_eval/curated/my_export \
        --backend ultralytics --weights ./weights/my_model.pt \
        --name my-model-v1 --out-dir /tmp/bakeoff

Run once per model; the per-model JSON files are then merged into the
comparison tables. Accuracy is identical-metric (COCOeval) across
backends; latency is reported per the model's native runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from .backends.factory import _build_backend, parse_class_ids
from .dataset import YoloTestSet
from .metrics import coco_eval, detections_to_coco, operating_point, percentiles
from .profile import BACKENDS, BakeoffProfile, resolve_profile
from .report import log_mlflow, weights_size_mb, write_report


if TYPE_CHECKING:
    from .backends.base import Detection, Detector


def _str2bool(value: str) -> bool:
    """Parse a CLI bool that may arrive as a string from the job runner."""
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _run_inference(
    backend: Detector, ds: YoloTestSet, *, warmup: int
) -> tuple[dict[int, list[Detection]], list[float]]:
    """Run the backend over every test image, timing each detect call."""
    dets_by_image: dict[int, list[Detection]] = {}
    latencies_ms: list[float] = []
    for idx, img in enumerate(ds.images):
        frame_bgr = cv2.imread(str(img.path))
        if frame_bgr is None:
            dets_by_image[img.image_id] = []
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        dets = backend.detect(frame_rgb)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        dets_by_image[img.image_id] = dets
        if idx >= warmup:  # drop warmup frames from latency stats
            latencies_ms.append(dt_ms)
    return dets_by_image, latencies_ms


def _per_stratum(
    ds: YoloTestSet, dets_by_image: dict[int, list[Detection]]
) -> dict[str, dict[str, float]]:
    """mAP@0.5 + operating-point recall per stratum (cluster), if mapped."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for img in ds.images:
        groups[ds.stratum_for(img)].append(img)
    out: dict[str, dict[str, float]] = {}
    for stratum, imgs in sorted(groups.items()):
        image_ids = {i.image_id for i in imgs}
        sub_gt = {
            'images': [
                {'id': i.image_id, 'file_name': i.path.name, 'width': i.width, 'height': i.height}
                for i in imgs
            ],
            'annotations': [],
            'categories': [{'id': 1, 'name': ds.target_class_name}],
        }
        ann_id = 1
        for i in imgs:
            for x1, y1, x2, y2 in i.boxes:
                sub_gt['annotations'].append(
                    {
                        'id': ann_id,
                        'image_id': i.image_id,
                        'category_id': 1,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': (x2 - x1) * (y2 - y1),
                        'iscrowd': 0,
                    }
                )
                ann_id += 1
        sub_dets = {iid: dets_by_image.get(iid, []) for iid in image_ids}
        m = coco_eval(sub_gt, detections_to_coco(sub_dets))
        op = operating_point(imgs, sub_dets)
        out[stratum] = {
            'n_images': len(imgs),
            'map_50': m.map_50,
            'recall': op.recall,
            'precision': op.precision,
        }
    return out


# argparse dest -> BakeoffProfile field. These flags default to None so an
# explicit CLI value can be told apart from "use the profile's value".
PROFILE_BACKED_ARGS: dict[str, str] = {
    'gt_class_id': 'target_class_id',
    'gt_class_name': 'target_class_name',
    'backend': 'default_backend',
    'imgsz': 'imgsz',
    'triton_model': 'triton_model',
    'primary_weights': 'context_weights',
    'primary_imgsz': 'context_imgsz',
    'primary_conf': 'context_conf',
    'conf_floor': 'conf_floor',
    'nms_iou': 'nms_iou',
    'op_conf': 'op_conf',
    'op_iou': 'op_iou',
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Detector bake-off (single backend run).')
    p.add_argument(
        '--profile',
        default=None,
        help='BakeoffProfile: registered/example name or a profile .json (default: generic)',
    )
    p.add_argument('--dataset', type=Path, help='Export root with images/<split> + labels/<split>')
    p.add_argument('--split', default='test')
    p.add_argument('--images', type=Path, help='Override: images dir (instead of --dataset)')
    p.add_argument('--labels', type=Path, help='Override: labels dir')
    p.add_argument('--stratum-map', type=Path, help='JSON {image_stem: stratum} for per-cluster')
    p.add_argument('--gt-class-id', type=int, default=None, help='Target class id in GT labels')
    p.add_argument(
        '--gt-class-name',
        default=None,
        help='Display name for the target class (cosmetic, COCO categories block)',
    )
    p.add_argument('--backend', default=None, choices=list(BACKENDS))
    p.add_argument('--weights', help='Model weights path (backend-specific)')
    p.add_argument('--name', help='Model display name for the report')
    p.add_argument('--imgsz', type=int, default=None)
    p.add_argument('--device', default='0')
    p.add_argument('--pred-class-id', type=int, default=None, help='Keep only this pred class')
    # full = detector on the source frame; crop = coarse->crop->detector
    # (a cascade deployment mode). Applies to ANY backend.
    p.add_argument('--mode', choices=['full', 'crop'], default='full')
    # LPDNet (NVIDIA TAO DetectNet_v2): a domain-specific public baseline
    # model; see backends/lpdnet.py for when it applies.
    p.add_argument('--lpdnet-variant', choices=['usa', 'ccpd'], default='usa')
    # Triton backend
    p.add_argument('--triton-url', default='localhost:4601')
    p.add_argument(
        '--triton-model', help='Triton model name to score (required for --backend triton)'
    )
    # ONNX Runtime / CoreML backends (the quantized portable artifacts).
    p.add_argument(
        '--ort-providers',
        default='CUDAExecutionProvider,CPUExecutionProvider',
        help='Comma-separated ONNX Runtime EPs in priority order',
    )
    p.add_argument(
        '--coords-normalized',
        type=_str2bool,
        default=False,
        help='True if the model outputs normalized [0,1] coords (False for our ultralytics ONNX)',
    )
    p.add_argument(
        '--coreml-compute-units', default='ALL', help='CoreML ComputeUnit (ALL/CPU_ONLY/...)'
    )
    # Coarse stage (crop mode + two-stage): a detector filtered to the
    # profile's context (parent) classes.
    p.add_argument('--primary-weights', default=None)
    p.add_argument(
        '--primary-classes',
        default=None,
        help="Comma-separated coarse-stage class ids (default: the profile's context_class_ids; "
        "'' keeps every class)",
    )
    p.add_argument('--primary-imgsz', type=int, default=None)
    p.add_argument('--primary-conf', type=float, default=None)
    p.add_argument('--secondary-backend', choices=['ultralytics', 'triton'], default='ultralytics')
    p.add_argument(
        '--secondary-imgsz', type=int, default=640, help='Crop input size (two-stage fine stage)'
    )
    p.add_argument('--conf-floor', type=float, default=None, help='Low floor so mAP sees full PR')
    p.add_argument('--nms-iou', type=float, default=None, help='NMS IoU during inference')
    p.add_argument('--op-conf', type=float, default=None, help='Operating-point confidence')
    p.add_argument('--op-iou', type=float, default=None, help='Operating-point match IoU')
    p.add_argument('--warmup', type=int, default=3, help='Frames excluded from latency stats')
    p.add_argument('--out-dir', type=Path, default=Path('/tmp/bakeoff'))
    p.add_argument('--training-data', help="Note on this model's training data (for the report)")
    # MLflow logging on by default (the MLflow tracking service on :5000); the
    # logger fails soft if the server/package is unavailable. Use --no-mlflow
    # to disable.
    p.add_argument(
        '--mlflow',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Log run to the MLflow service for comparison charts',
    )
    p.add_argument(
        '--mlflow-uri',
        default=os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
    )
    p.add_argument('--mlflow-experiment', default='bakeoff')
    return p


def resolve_args(args: argparse.Namespace) -> tuple[argparse.Namespace, BakeoffProfile]:
    """Fill every flag the user left unset from the profile; validate.

    Raises ``SystemExit`` for an unknown profile or a triton backend with
    no model (the profile's ``triton_model`` is empty by design).
    """
    try:
        profile = resolve_profile(args.profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for dest, attr in PROFILE_BACKED_ARGS.items():
        if getattr(args, dest) is None:
            setattr(args, dest, getattr(profile, attr))
    if args.primary_classes is None:
        args.primary_classes = ','.join(str(c) for c in profile.context_class_ids)
    args.profile_name = profile.name
    if not args.triton_model:
        args.triton_model = None
    if args.backend == 'triton' and not args.triton_model:
        raise SystemExit("--triton-model (or the profile's triton_model) is required for triton")
    if args.backend == 'two-stage' and args.secondary_backend == 'triton' and not args.triton_model:
        raise SystemExit('--triton-model is required when --secondary-backend triton')
    return args, profile


def main(argv: list[str] | None = None) -> int:
    args, _profile = resolve_args(build_parser().parse_args(argv))

    if args.images and args.labels:
        ds = YoloTestSet(
            args.images,
            args.labels,
            target_class_id=args.gt_class_id,
            target_class_name=args.gt_class_name,
            stratum_map=(json.loads(args.stratum_map.read_text()) if args.stratum_map else None),
        )
    elif args.dataset:
        ds = YoloTestSet.from_dataset_root(
            args.dataset,
            split=args.split,
            target_class_id=args.gt_class_id,
            target_class_name=args.gt_class_name,
            stratum_map_path=args.stratum_map,
        )
    else:
        raise SystemExit('provide --dataset or both --images and --labels')

    if not ds.images:
        raise SystemExit(f'no images found under {args.images or args.dataset}')

    print(
        f'test set: {len(ds.images)} frames '
        f'({ds.n_positive_frames} positive, {ds.n_background_frames} background)'
    )

    backend = _build_backend(args)
    print(f'backend: {backend.name} [{backend.runtime}] imgsz={args.imgsz}')

    dets_by_image, latencies = _run_inference(backend, ds, warmup=args.warmup)

    metrics = coco_eval(ds.coco_gt(), detections_to_coco(dets_by_image))
    op = operating_point(ds.images, dets_by_image, conf=args.op_conf, iou=args.op_iou)
    lat = percentiles(latencies)
    throughput = (1000.0 / lat['mean']) if lat['mean'] > 0 else 0.0
    strata = _per_stratum(ds, dets_by_image) if ds.stratum_map else {}

    report: dict[str, Any] = {
        'model': backend.name,
        'runtime': backend.runtime,
        'profile': args.profile_name,
        'target_class': {'id': args.gt_class_id, 'name': args.gt_class_name},
        'imgsz': args.imgsz,
        'test_frames': len(ds.images),
        'positive_frames': ds.n_positive_frames,
        'background_frames': ds.n_background_frames,
        'coco': asdict(metrics),
        'operating_point': asdict(op),
        'latency_ms': lat,
        'throughput_fps': throughput,
        'per_stratum': strata,
    }

    size_mb = weights_size_mb(args.weights)
    if size_mb is not None:
        report['size_mb'] = size_mb

    if args.training_data:
        report['training_data'] = args.training_data

    safe = backend.name.replace('/', '_').replace(' ', '_')
    write_report(args.out_dir, safe, report)

    if args.mlflow:
        log_mlflow(report, args)

    print(
        f'\n{backend.name}\n'
        f'  mAP@.5:.95 {metrics.map_50_95:.4f} | mAP@.5 {metrics.map_50:.4f} | '
        f'mAP@.75 {metrics.map_75:.4f} | AP_small {metrics.ap_small:.4f}\n'
        f'  op(conf={op.conf},iou={op.iou})  P {op.precision:.4f}  R {op.recall:.4f}  '
        f'F1 {op.f1:.4f}  meanIoU {op.mean_iou:.4f}  (tp={op.tp} fp={op.fp} fn={op.fn})\n'
        f'  latency mean {lat["mean"]:.1f} ms (p90 {lat["p90"]:.1f})  ~{throughput:.1f} fps'
    )
    return 0


__all__ = ['build_parser', 'main', 'parse_class_ids', 'resolve_args']


if __name__ == '__main__':
    raise SystemExit(main())

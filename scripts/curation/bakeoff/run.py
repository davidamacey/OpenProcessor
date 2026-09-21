"""LPR bake-off CLI: score one detector backend on the frozen test split.

Example:
    .venv/bin/python -m scripts.curation.bakeoff.run \
        --dataset ./data/bakeoff_eval/curated/lpr_current \
        --backend ultralytics --weights ./weights/lpr_nanov11_640.pt \
        --imgsz 1280 --name lpr-nanov11-640 --out-dir /tmp/bakeoff

Run once per model; the per-model JSON files are then merged into the
comparison tables for the paper. Accuracy is identical-metric (COCOeval)
across backends; latency is reported per the model's native runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from .dataset import YoloTestSet
from .metrics import coco_eval, detections_to_coco, operating_point, percentiles


if TYPE_CHECKING:
    from .backends.base import Detection, Detector


def _str2bool(value: str) -> bool:
    """Parse a CLI bool that may arrive as a string from the job runner."""
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _weights_size_mb(weights: str | None) -> float | None:
    """On-disk size of a weights file/dir in MB (recursive for .mlpackage dirs)."""
    if not weights:
        return None
    path = Path(weights)
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 3)
    if path.is_dir():  # CoreML .mlpackage is a directory
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return round(total / (1024 * 1024), 3)
    return None


def _build_detector(args: argparse.Namespace, backend: str) -> Detector:
    """Build ONE model detector (the model under test) for a backend."""
    if backend == 'ultralytics':
        from .backends.ultralytics_pt import UltralyticsDetector

        return UltralyticsDetector(
            args.weights,
            name=args.name,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf_floor,
            iou=args.nms_iou,
            plate_class_id=args.plate_class_id,
        )
    if backend == 'triton':
        from .backends.triton_trt import TritonLprDetector

        return TritonLprDetector(
            url=args.triton_url,
            model=args.triton_model,
            name=args.name,
            input_size=args.imgsz,
            conf=args.conf_floor,
            iou=args.nms_iou,
        )
    if backend == 'open-image-models':
        from .backends.open_image_models import OpenImageModelsDetector

        return OpenImageModelsDetector(
            name=args.name,
            conf=args.conf_floor,
            device='cpu' if str(args.device) == 'cpu' else 'cuda',
        )
    if backend == 'lpdnet':
        from .backends.lpdnet import LpdnetDetector

        return LpdnetDetector(
            args.weights,
            name=args.name,
            variant=args.lpdnet_variant,
            conf=args.conf_floor,
            iou=args.nms_iou,
            device='cpu' if str(args.device) == 'cpu' else 'cuda',
        )
    if backend == 'onnxruntime':
        from .backends.onnxruntime_ort import OnnxRuntimeDetector

        return OnnxRuntimeDetector(
            args.weights,
            name=args.name,
            imgsz=args.imgsz,
            conf=args.conf_floor,
            iou=args.nms_iou,
            providers=args.ort_providers,
            coords_normalized=args.coords_normalized,
        )
    if backend == 'coreml':
        from .backends.coreml import CoreMLDetector

        return CoreMLDetector(
            args.weights,
            name=args.name,
            imgsz=args.imgsz,
            conf=args.conf_floor,
            iou=args.nms_iou,
            compute_units=args.coreml_compute_units,
            coords_normalized=args.coords_normalized,
        )
    raise SystemExit(f'unknown / not-yet-wired backend: {backend!r}')


def _vehicle_detector(args: argparse.Namespace) -> Detector:
    """COCO vehicle detector for crop mode (keeps car/motorcycle/bus/truck)."""
    from .backends.ultralytics_pt import UltralyticsDetector

    keep = {int(c) for c in str(args.vehicle_classes).split(',') if c.strip()}
    return UltralyticsDetector(
        args.vehicle_weights,
        name='vehicle',
        imgsz=args.vehicle_imgsz,
        device=args.device,
        conf=args.vehicle_conf,
        iou=args.nms_iou,
        keep_classes=keep or None,
    )


def _build_backend(args: argparse.Namespace) -> Detector:
    """Build the system under test, honoring --mode (full vs crop).

    ``--mode crop`` wraps ANY backend in the vehicle->crop->detector
    pipeline (the sorter deployment mode); ``--mode full`` runs the
    detector directly on the source frame. The legacy ``two-stage`` backend
    remains for the bespoke in-house vehicle+crop-LPR combo.
    """
    if args.backend == 'two-stage':
        from .backends.two_stage import TwoStageDetector
        from .backends.ultralytics_pt import UltralyticsDetector

        vehicle = _vehicle_detector(args)
        if args.lpr_backend == 'triton':
            from .backends.triton_trt import TritonLprDetector

            lpr: Detector = TritonLprDetector(
                url=args.triton_url,
                model=args.triton_model,
                input_size=640,
                conf=args.conf_floor,
                iou=args.nms_iou,
            )
        else:
            lpr = UltralyticsDetector(
                args.weights,
                name='lpr',
                imgsz=args.lpr_imgsz,
                device=args.device,
                conf=args.conf_floor,
                iou=args.nms_iou,
            )
        return TwoStageDetector(
            vehicle, lpr, name=args.name or 'in-house 2-stage', nms_iou=args.nms_iou
        )

    inner = _build_detector(args, args.backend)
    if args.mode == 'crop':
        from .backends.two_stage import TwoStageDetector

        return TwoStageDetector(
            _vehicle_detector(args),
            inner,
            name=args.name or inner.name,
            vehicle_conf=args.vehicle_conf,
            nms_iou=args.nms_iou,
        )
    return inner


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
            'categories': [{'id': 1, 'name': 'license_plate'}],
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


def main() -> int:
    p = argparse.ArgumentParser(description='LPR detector bake-off (single backend run).')
    p.add_argument('--dataset', type=Path, help='Export root with images/<split> + labels/<split>')
    p.add_argument('--split', default='test')
    p.add_argument('--images', type=Path, help='Override: images dir (instead of --dataset)')
    p.add_argument('--labels', type=Path, help='Override: labels dir')
    p.add_argument('--stratum-map', type=Path, help='JSON {image_stem: stratum} for per-cluster')
    p.add_argument('--gt-plate-class', type=int, default=0, help='Plate class id in GT labels')
    p.add_argument(
        '--backend',
        required=True,
        choices=[
            'ultralytics',
            'triton',
            'open-image-models',
            'two-stage',
            'lpdnet',
            'onnxruntime',
            'coreml',
        ],
    )
    p.add_argument('--weights', help='Model weights path (backend-specific)')
    p.add_argument('--name', help='Model display name for the report')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--device', default='0')
    p.add_argument('--plate-class-id', type=int, default=None, help='Keep only this pred class')
    # full = detector on the source frame; crop = vehicle->crop->detector
    # (the sorter deployment mode). Applies to ANY backend.
    p.add_argument('--mode', choices=['full', 'crop'], default='full')
    # LPDNet (NVIDIA TAO DetectNet_v2) backend
    p.add_argument('--lpdnet-variant', choices=['usa', 'ccpd'], default='usa')
    # Triton backend
    p.add_argument('--triton-url', default='localhost:4601')
    p.add_argument('--triton-model', default='lpr_nanov11_640')
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
    # Vehicle stage (crop mode + two-stage). A COCO detector (YOLO11/YOLO26)
    # filtered to vehicle classes (car=2, motorcycle=3, bus=5, truck=7).
    p.add_argument('--vehicle-weights', default='./weights/yolo11n.pt')
    p.add_argument('--vehicle-classes', default='2,3,5,7', help='COCO vehicle class ids to keep')
    p.add_argument('--vehicle-imgsz', type=int, default=960)
    p.add_argument('--vehicle-conf', type=float, default=0.25)
    p.add_argument('--lpr-backend', choices=['ultralytics', 'triton'], default='ultralytics')
    p.add_argument('--lpr-imgsz', type=int, default=640, help='Crop LPR input size (two-stage)')
    p.add_argument('--conf-floor', type=float, default=0.001, help='Low floor so mAP sees full PR')
    p.add_argument('--nms-iou', type=float, default=0.7, help='NMS IoU during inference')
    p.add_argument('--op-conf', type=float, default=0.25, help='Operating-point confidence')
    p.add_argument('--op-iou', type=float, default=0.45, help='Operating-point match IoU')
    p.add_argument('--warmup', type=int, default=3, help='Frames excluded from latency stats')
    p.add_argument('--out-dir', type=Path, default=Path('/tmp/lpr_bakeoff'))
    p.add_argument('--training-data', help="Note on this model's training data (for the paper)")
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
    p.add_argument('--mlflow-experiment', default='lpr-bakeoff')
    args = p.parse_args()

    if args.images and args.labels:
        ds = YoloTestSet(
            args.images,
            args.labels,
            plate_class_id=args.gt_plate_class,
            stratum_map=(json.loads(args.stratum_map.read_text()) if args.stratum_map else None),
        )
    elif args.dataset:
        ds = YoloTestSet.from_dataset_root(
            args.dataset,
            split=args.split,
            plate_class_id=args.gt_plate_class,
            stratum_map_path=args.stratum_map,
        )
    else:
        raise SystemExit('provide --dataset or both --images and --labels')

    if not ds.images:
        raise SystemExit(f'no images found under {args.images or args.dataset}')

    print(
        f'test set: {len(ds.images)} frames '
        f'({ds.n_positive_frames} with plates, {ds.n_background_frames} background)'
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

    size_mb = _weights_size_mb(args.weights)
    if size_mb is not None:
        report['size_mb'] = size_mb

    if args.training_data:
        report['training_data'] = args.training_data

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe = backend.name.replace('/', '_').replace(' ', '_')
    (args.out_dir / f'{safe}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    _write_summary_row(args.out_dir / 'summary.csv', report)

    if args.mlflow:
        _log_mlflow(report, args)

    print(
        f'\n{backend.name}\n'
        f'  mAP@.5:.95 {metrics.map_50_95:.4f} | mAP@.5 {metrics.map_50:.4f} | '
        f'mAP@.75 {metrics.map_75:.4f} | AP_small {metrics.ap_small:.4f}\n'
        f'  op(conf={op.conf},iou={op.iou})  P {op.precision:.4f}  R {op.recall:.4f}  '
        f'F1 {op.f1:.4f}  meanIoU {op.mean_iou:.4f}  (tp={op.tp} fp={op.fp} fn={op.fn})\n'
        f'  latency mean {lat["mean"]:.1f} ms (p90 {lat["p90"]:.1f})  ~{throughput:.1f} fps'
    )
    return 0


def _log_mlflow(report: dict[str, Any], args: argparse.Namespace) -> None:
    """Optionally log this run to MLflow so its UI charts the comparison.

    Flag-gated and import-guarded: if mlflow isn't installed we just skip,
    keeping it off the harness's hard dependencies.
    """
    try:
        import mlflow
    except ImportError:
        print('mlflow not installed; skipping (.venv/bin/pip install mlflow)')
        return
    try:
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=report['model']):
            mlflow.log_params(
                {
                    'model': report['model'],
                    'runtime': report['runtime'],
                    'imgsz': report['imgsz'],
                    'training_data': args.training_data or 'unknown',
                }
            )
            c, op, lat = report['coco'], report['operating_point'], report['latency_ms']
            mlflow.log_metrics(
                {
                    'map_50_95': c['map_50_95'],
                    'map_50': c['map_50'],
                    'map_75': c['map_75'],
                    'ap_small': c['ap_small'],
                    'ap_medium': c['ap_medium'],
                    'ap_large': c['ap_large'],
                    'precision': op['precision'],
                    'recall': op['recall'],
                    'f1': op['f1'],
                    'mean_iou': op['mean_iou'],
                    'latency_mean_ms': lat['mean'],
                    'throughput_fps': report['throughput_fps'],
                }
            )
            for stratum, vals in report.get('per_stratum', {}).items():
                key = ''.join(ch if ch.isalnum() or ch in '-_./' else '_' for ch in stratum)
                mlflow.log_metric(f'stratum_map50/{key}', float(vals['map_50']))
            # Artifact upload is best-effort: it needs the server's
            # --serve-artifacts proxy and can fail on artifact-root perms
            # without invalidating the (already-committed) metrics.
            try:
                safe = report['model'].replace('/', '_').replace(' ', '_')
                mlflow.log_dict(report, f'{safe}.json')
            except Exception as art_exc:
                print(f'mlflow artifact upload skipped: {art_exc}')
        print(f'logged to mlflow: {args.mlflow_uri} (experiment={args.mlflow_experiment})')
    except Exception as exc:  # server unreachable / transient — don't fail the run
        print(f'mlflow logging skipped: {exc}')


def _write_summary_row(path: Path, report: dict[str, Any]) -> None:
    """Append one row to a shared summary.csv so models accumulate."""
    fields = [
        'model',
        'runtime',
        'imgsz',
        'map_50_95',
        'map_50',
        'map_75',
        'ap_small',
        'precision',
        'recall',
        'f1',
        'mean_iou',
        'latency_mean_ms',
        'throughput_fps',
    ]
    row = {
        'model': report['model'],
        'runtime': report['runtime'],
        'imgsz': report['imgsz'],
        'map_50_95': round(report['coco']['map_50_95'], 4),
        'map_50': round(report['coco']['map_50'], 4),
        'map_75': round(report['coco']['map_75'], 4),
        'ap_small': round(report['coco']['ap_small'], 4),
        'precision': round(report['operating_point']['precision'], 4),
        'recall': round(report['operating_point']['recall'], 4),
        'f1': round(report['operating_point']['f1'], 4),
        'mean_iou': round(report['operating_point']['mean_iou'], 4),
        'latency_mean_ms': round(report['latency_ms']['mean'], 2),
        'throughput_fps': round(report['throughput_fps'], 2),
    }
    exists = path.is_file()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


if __name__ == '__main__':
    raise SystemExit(main())

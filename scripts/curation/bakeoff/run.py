"""Detector bake-off CLI: score one detector backend on a frozen test split.

Domain-agnostic: point it at any YOLO test split and any of the wired
backends. Every class present in the split is scored (a profile's
``class_filter`` narrows it); the model's classes are mapped to the eval
classes either by the map the job carries (``--class-map-json``) or by
name. What else is measured -- the context classes a crop-mode cascade
crops on, the Triton model, backend and metric defaults -- comes from a
:class:`~scripts.curation.bakeoff.profile.BakeoffProfile` (``--profile``;
the neutral ``generic`` profile when omitted). Any explicit CLI flag
overrides the profile's value.

Example:
    .venv/bin/python -m scripts.curation.bakeoff.run \
        --dataset ./data/exports/<export> \
        --backend ultralytics --weights ./weights/my_model.pt \
        --model-key custom:my-model --out-dir /tmp/bakeoff

Run once per model; the per-model JSON reports (schema v2) are then merged
into the comparison by ``compare.py``. Accuracy is identical-metric
(COCOeval) across backends; latency is reported per the model's native
runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from . import class_map as cmap, scoring
from .backends.factory import _build_backend, parse_class_ids
from .dataset import YoloTestSet
from .metrics import percentiles
from .profile import BakeoffProfile, resolve_profile
from .report import build_report, log_mlflow, weights_size_mb, write_report


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


# argparse dest -> BakeoffProfile field. These flags default to None so an
# explicit CLI value can be told apart from "use the profile's value".
PROFILE_BACKED_ARGS: dict[str, str] = {
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

# backend_options keys that set a built-in backend's CLI flag (argparse dest).
# Every other key stays in ``args.backend_options`` for plugin backends.
BACKEND_OPTION_ARGS: dict[str, str] = {
    'providers': 'ort_providers',
    'coords_normalized': 'coords_normalized',
    'compute_units': 'coreml_compute_units',
    'triton_url': 'triton_url',
    'primary_weights': 'primary_weights',
    'primary_classes': 'primary_classes',
    'primary_imgsz': 'primary_imgsz',
    'primary_conf': 'primary_conf',
    'secondary_backend': 'secondary_backend',
    'secondary_imgsz': 'secondary_imgsz',
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Detector bake-off (single backend run).')
    p.add_argument(
        '--profile',
        default=None,
        help='BakeoffProfile: registered name or a profile .json path (default: generic)',
    )
    p.add_argument('--dataset', type=Path, help='Export root with images/<split> + labels/<split>')
    p.add_argument('--split', default='test')
    p.add_argument('--images', type=Path, help='Override: images dir (instead of --dataset)')
    p.add_argument('--labels', type=Path, help='Override: labels dir')
    p.add_argument('--stratum-map', type=Path, help='JSON {image_stem: stratum} for per-cluster')
    p.add_argument('--backend', default=None, help='Built-in or plugin-registered backend')
    p.add_argument('--weights', help='Model weights path (backend-specific)')
    p.add_argument('--name', help='Backend display name (default: --display-name)')
    p.add_argument('--model-key', help='Unique model key in the job (report file stem)')
    p.add_argument('--display-name', help='Human label for the report (default: model key)')
    p.add_argument('--source', default='custom', help='run | baseline | custom')
    p.add_argument('--run-id', default=None, help='Training run id (source=run)')
    p.add_argument(
        '--class-map-json',
        default=None,
        help='{"<model_class_id>": <eval_class_id>} (or a full class-mapping object); '
        "omitted/null = match the loaded model's class names",
    )
    p.add_argument('--backend-options-json', default='{}', help='Backend-specific options object')
    p.add_argument(
        '--train-test-overlap-json',
        default='null',
        help='{"n_images", "fraction"} computed at enqueue (null if unknown)',
    )
    p.add_argument('--imgsz', type=int, default=None)
    p.add_argument('--device', default='0')
    # full = detector on the source frame; crop = coarse->crop->detector
    # (a cascade deployment mode). Applies to ANY backend.
    p.add_argument('--mode', choices=['full', 'crop'], default='full')
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
        help='True if the model outputs normalized [0,1] coords (False for Ultralytics ONNX)',
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


def _json_arg(raw: str | None, flag: str) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise SystemExit(f'{flag}: invalid JSON ({exc})') from exc


def _apply_class_map(args: argparse.Namespace) -> None:
    given = _json_arg(args.class_map_json, '--class-map-json')
    if given is not None and not isinstance(given, dict):
        raise SystemExit('--class-map-json must be an object or null')
    args.class_map_given = given
    raw = (given or {}).get('model_to_eval', given) if given else None
    args.class_map = {int(k): int(v) for k, v in raw.items()} if raw else None


def _apply_backend_options(args: argparse.Namespace) -> None:
    opts = _json_arg(args.backend_options_json, '--backend-options-json') or {}
    if not isinstance(opts, dict):
        raise SystemExit('--backend-options-json must be an object')
    args.backend_options = opts
    for key, dest in BACKEND_OPTION_ARGS.items():
        if key not in opts:
            continue
        value = opts[key]
        if key == 'primary_classes' and isinstance(value, list):
            value = ','.join(str(v) for v in value)
        setattr(args, dest, value)


def resolve_args(args: argparse.Namespace) -> tuple[argparse.Namespace, BakeoffProfile]:
    """Fill every flag the user left unset from the profile; parse JSON flags; validate.

    Raises ``SystemExit`` for an unknown profile, malformed JSON flags, or a
    triton backend with no model (the profile's ``triton_model`` is empty by
    design).
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
    _apply_class_map(args)
    _apply_backend_options(args)
    args.train_test_overlap = _json_arg(args.train_test_overlap_json, '--train-test-overlap-json')
    args.model_key = args.model_key or args.name or args.backend
    args.display_name = args.display_name or args.name or args.model_key
    args.name = args.name or args.display_name
    args.profile_name = profile.name
    if not args.triton_model:
        args.triton_model = None
    if args.backend == 'triton' and not args.triton_model:
        raise SystemExit("--triton-model (or the profile's triton_model) is required for triton")
    if args.backend == 'two-stage' and args.secondary_backend == 'triton' and not args.triton_model:
        raise SystemExit('--triton-model is required when --secondary-backend triton')
    return args, profile


def report_stem(model_key: str) -> str:
    """Per-model report file stem: the model key with unsafe characters -> ``_``."""
    return re.sub(r'[^A-Za-z0-9_.-]', '_', model_key)


def _load_dataset(args: argparse.Namespace) -> YoloTestSet:
    stratum = args.stratum_map
    if args.images and args.labels:
        return YoloTestSet(
            args.images,
            args.labels,
            class_names=cmap.read_names(args.labels.parent.parent / 'data.yaml'),
            stratum_map=(json.loads(stratum.read_text()) if stratum else None),
        )
    if args.dataset:
        return YoloTestSet.from_dataset_root(
            args.dataset, split=args.split, stratum_map_path=stratum
        )
    raise SystemExit('provide --dataset or both --images and --labels')


def scored_classes(ds: YoloTestSet, profile: BakeoffProfile) -> list[int]:
    """Classes present in the split, narrowed by the profile's ``class_filter`` (names)."""
    present = sorted(ds.present_class_ids)
    if not profile.class_filter:
        return present
    wanted = {cmap.normalize_name(n) for n in profile.class_filter}
    return [c for c in present if cmap.normalize_name(ds.class_names[c]) in wanted]


def resolve_mapping(
    args: argparse.Namespace, backend: Detector, ds: YoloTestSet, scored: list[int]
) -> cmap.ClassMapping:
    model_names = getattr(backend, 'class_names', None)
    if args.class_map_given is not None:
        return cmap.from_given(
            args.class_map_given,
            model_names=model_names,
            eval_names=ds.class_names,
            scored_class_ids=scored,
        )
    return cmap.resolve_for_loaded_model(model_names, ds.class_names, scored)


def main(argv: list[str] | None = None) -> int:
    args, profile = resolve_args(build_parser().parse_args(argv))

    ds = _load_dataset(args)
    if not ds.images:
        raise SystemExit(f'no images found under {args.images or args.dataset}')
    scored = scored_classes(ds, profile)
    print(
        f'test set: {len(ds.images)} frames '
        f'({ds.n_positive_frames} positive, {ds.n_background_frames} background), '
        f'{len(scored)} scored classes'
    )

    backend = _build_backend(args)
    print(f'backend: {backend.name} [{backend.runtime}] imgsz={args.imgsz}')
    mapping = resolve_mapping(args, backend, ds, scored)

    dets_by_image, latencies = _run_inference(backend, ds, warmup=args.warmup)
    blocks = scoring.score(
        ds, dets_by_image, mapping, scored_class_ids=scored, conf=args.op_conf, iou=args.op_iou
    )
    report = build_report(
        args,
        runtime=backend.runtime,
        blocks=blocks,
        mapping=mapping,
        latency_ms=percentiles(latencies),
        size_mb=weights_size_mb(args.weights),
        frames=(len(ds.images), ds.n_positive_frames, ds.n_background_frames),
    )
    write_report(args.out_dir, report_stem(args.model_key), report)
    if args.mlflow:
        log_mlflow(report, args)

    o, lat = report['overall'], report['latency_ms']
    print(
        f'\n{args.display_name}  [{mapping.method}] '
        f'{report["coverage"]["n_covered"]}/{report["coverage"]["n_eval_classes"]} classes\n'
        f'  mAP@.5:.95 {o["map_50_95"]}  mAP@.5 {o["map_50"]}  '
        f'P {o["precision"]}  R {o["recall"]}  F1 {o["f1"]}  '
        f'(tp={o["tp"]} fp={o["fp"]} fn={o["fn"]})\n'
        f'  latency mean {lat["mean"]:.1f} ms (p90 {lat["p90"]:.1f})  ~{report["fps"]:.1f} fps'
    )
    return 0


__all__ = [
    'build_parser',
    'main',
    'parse_class_ids',
    'report_stem',
    'resolve_args',
    'scored_classes',
]


if __name__ == '__main__':
    raise SystemExit(main())

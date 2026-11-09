#!/usr/bin/env python3
"""One export for every Ultralytics detection family: a fused ``output0 [batch, 300, 6]``.

``YOLO(pt).export(format='onnx', ...)`` with the ``nms`` argument selects the head that is served
(https://docs.ultralytics.com/modes/export/): the NMS-free one-to-one head of YOLO26 and YOLOv10
(``nms=False``), or NMS embedded in the graph for every other family (``nms=True``). Both end in
the same tensor of ``[x1, y1, x2, y2, score, class]`` rows in letterboxed input pixels, so one
engine contract serves them all (``src/domain/model_families.py`` holds the recipes). A graph that
is not that tensor is refused, never built.

Steps: ONNX export, head check, ModelOpt FP16 bake and uint8 input, TensorRT engine
(``yolo_trt_build.py``), ``config.pbtxt`` (``yolo_triton_config.py``) and ``labels.txt``.

Formats: ``onnx`` stops after the head check; ``trt`` (also ``all``) builds the engine.

Usage::

    docker compose run --rm model-export python /app/export/ultralytics_export.py --models small
    docker compose run --rm model-export python /app/export/ultralytics_export.py \\
        --custom-model /app/pytorch_models/my_model.pt:my_detector:32
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
from trt_build_settings import max_batch as engine_max_batch
from yolo_export_report import any_export_failed, print_summary, save_results_file
from yolo_export_settings import DEVICE, IMG_SIZE
from yolo_model_config import (
    extract_class_names,
    load_config_file,
    parse_custom_model,
    save_labels_file,
)
from yolo_triton_config import save_fused_config

from src.domain.model_families import MAX_DET, ONNX_SUFFIX, engine_name, export_kwargs, head_kind_of


if TYPE_CHECKING:
    from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path('/app/models')
FORMATS = ('onnx', 'trt', 'all')

# Built-in models: YOLO11 sizes (the default detector is ``small``) and YOLO26 sizes. The
# per-size batch keeps the batch-64 FP16 activation footprint off 12 GB-class cards; raise it on
# bigger GPUs through ``--custom-model``.
DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    'nano': {'pt_file': '/app/pytorch_models/yolo11n.pt', 'triton_name': 'yolov11_nano', 'max_batch': 128},
    'small': {'pt_file': '/app/pytorch_models/yolo11s.pt', 'triton_name': 'yolov11_small', 'max_batch': 64},
    'medium': {'pt_file': '/app/pytorch_models/yolo11m.pt', 'triton_name': 'yolov11_medium', 'max_batch': 32},
    'large': {'pt_file': '/app/pytorch_models/yolo11l.pt', 'triton_name': 'yolov11_large', 'max_batch': 16},
    'xlarge': {'pt_file': '/app/pytorch_models/yolo11x.pt', 'triton_name': 'yolov11_xlarge', 'max_batch': 8},
    'yolo26_nano': {'pt_file': '/app/pytorch_models/yolo26n.pt', 'triton_name': 'yolo26_nano', 'max_batch': 128},
    'yolo26_small': {'pt_file': '/app/pytorch_models/yolo26s.pt', 'triton_name': 'yolo26_small', 'max_batch': 32},
    'yolo26_medium': {'pt_file': '/app/pytorch_models/yolo26m.pt', 'triton_name': 'yolo26_medium', 'max_batch': 32},
    'yolo26_large': {'pt_file': '/app/pytorch_models/yolo26l.pt', 'triton_name': 'yolo26_large', 'max_batch': 16},
    'yolo26_xlarge': {'pt_file': '/app/pytorch_models/yolo26x.pt', 'triton_name': 'yolo26_xlarge', 'max_batch': 8},
}  # fmt: skip


def load_model(pt_file: str) -> YOLO:
    """The Ultralytics model of a ``.pt`` file."""
    from ultralytics import YOLO

    return YOLO(pt_file)


def validate_pt_file(pt_file: str) -> str | None:
    """Why ``pt_file`` cannot be exported, or ``None`` when it is a non-empty file."""
    path = Path(pt_file)
    if not path.is_file():
        return f'Model file not found: {pt_file}'
    if path.stat().st_size == 0:
        return f'Model file is empty: {pt_file}'
    return None


def inspect_fused_output(onnx_path: Path) -> tuple[str, list[int]]:
    """Return the fused output tensor's (name, per-item dims).

    The fused export emits exactly one output shaped ``(batch, max_det, 6)``; the per-item dims
    (without the batch axis) feed straight into config.pbtxt. Anything else is refused: the raw
    one-to-many head would be served and decoded as if it were NMS-free.
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
    if dims != [MAX_DET, 6]:
        raise RuntimeError(
            f'Output {out.name!r} has per-item dims {dims}, expected [{MAX_DET}, 6]: raw '
            'one-to-many head: the export did not select the NMS-free head (nms=False)'
        )
    logger.info(f'Fused output: name={out.name} per-item dims={dims}')
    return out.name, dims


def export_fused_onnx(
    model: YOLO, triton_name: str, max_batch: int
) -> tuple[Path, str, tuple[str, list[int]]]:
    """Export the fused ONNX to ``{triton_name}_end2end/1/model.onnx``.

    Returns the path, the model's head kind and the checked output (name, per-item dims). The
    export ``batch`` is ``max_batch``: with NMS in the graph it is a hard cap on the images
    answered per call (the NMS is unrolled for it).
    """
    kind = head_kind_of(model.model.model[-1])  # type: ignore[union-attr]  # ultralytics types YOLO.model as str | Module | None
    logger.info(f'Exporting fused ONNX ({kind} head, imgsz={IMG_SIZE}, batch={max_batch})...')
    exported = Path(
        model.export(
            format='onnx',
            imgsz=IMG_SIZE,
            dynamic=True,
            simplify=True,
            device=DEVICE,
            **export_kwargs(kind, max_batch),
        )
    )
    target = MODELS_DIR / f'{triton_name}{ONNX_SUFFIX}' / '1' / 'model.onnx'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, target)
    logger.info(f'ONNX saved: {target}')
    return target, kind, inspect_fused_output(target)


def export_model(model_id: str, config: dict[str, Any], formats: list[str]) -> dict[str, Any]:
    """Export one model; the result carries ``status`` ``success`` or ``error`` with its reason."""
    triton_name = config['triton_name']
    result: dict[str, Any] = {'model': model_id, 'triton_name': triton_name}
    logger.info('=' * 70)
    logger.info(f'Processing {model_id} -> {triton_name}')
    logger.info('=' * 70)
    problem = validate_pt_file(config['pt_file'])
    if problem is not None:
        logger.error(problem)
        return {**result, 'status': 'error', 'error': problem}

    max_batch = engine_max_batch('yolo', config['max_batch'])
    result['max_batch'] = max_batch
    try:
        model = load_model(config['pt_file'])
        class_names = config.get('class_names') or extract_class_names(model)
        onnx_path, kind, (output_name, output_dims) = export_fused_onnx(
            model, triton_name, max_batch
        )
        result.update(head_kind=kind, onnx=str(onnx_path))
        if 'trt' in formats or 'all' in formats:
            from yolo_trt_build import build_fused_engine

            engine = engine_name(triton_name, kind)
            model_dir = MODELS_DIR / engine
            plan_path = model_dir / '1' / 'model.plan'
            if not build_fused_engine(onnx_path, plan_path, max_batch):
                return {**result, 'status': 'error', 'error': 'Failed to build the engine'}
            save_fused_config(
                model_dir,
                engine,
                max_batch,
                output_name,
                output_dims,
                overwrite=config.get('overwrite_config', False),
            )
            save_labels_file(class_names, model_dir)
            result['trt'] = str(plan_path)
    except (RuntimeError, ValueError, OSError) as e:  # the failure is this model's result
        logger.exception(f'Export of {model_id} failed')
        return {**result, 'status': 'error', 'error': str(e)}
    return {**result, 'status': 'success'}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Export Ultralytics detection models to a fused Triton engine'
    )
    parser.add_argument(
        '--formats', nargs='+', choices=FORMATS, default=['trt'], help='onnx or trt (default: trt)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help=f'Built-in models: {", ".join(DEFAULT_MODELS)} (default: small)',
    )
    parser.add_argument(
        '--custom-model',
        nargs='+',
        dest='custom_models',
        metavar='PATH[:NAME][:BATCH]',
        help='Custom model(s): path/to/model.pt[:triton_name][:max_batch]',
    )
    parser.add_argument(
        '--config-file',
        metavar='PATH',
        help='YAML file with model definitions (export/models_config.example.yaml)',
    )
    parser.add_argument(
        '--overwrite-config',
        action='store_true',
        help='Replace an existing config.pbtxt that differs (default: write config.pbtxt.generated '
        'beside it)',
    )
    parser.add_argument('--list-models', action='store_true', help='List built-in models and exit')
    parser.add_argument('--verbose', '-v', action='store_true', help='DEBUG logging')
    return parser


def _select_models(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """The models to export from the config file, ``--models`` and ``--custom-model``."""
    selected: dict[str, dict[str, Any]] = {}
    try:
        if args.config_file:
            selected.update(load_config_file(args.config_file))
        for model_id in args.models or []:
            if model_id not in DEFAULT_MODELS:
                raise ValueError(
                    f"Unknown model '{model_id}'. Available: {', '.join(DEFAULT_MODELS)}. "
                    'Use --custom-model for custom models.'
                )
            selected[model_id] = dict(DEFAULT_MODELS[model_id])
        for custom in args.custom_models or []:
            model_id, config = parse_custom_model(custom)
            selected[model_id] = config
    except (FileNotFoundError, ValueError) as e:
        logger.error(f'Model selection: {e}')
        sys.exit(1)
    if not selected:
        logger.info('No models specified, defaulting to small')
        selected['small'] = dict(DEFAULT_MODELS['small'])
    return selected


def main() -> int:
    """Export the selected models; the exit code is 1 when any of them failed."""
    args = build_parser().parse_args()
    if args.list_models:
        for model_id, cfg in DEFAULT_MODELS.items():
            print(f'  {model_id:14} -> {cfg["triton_name"]:18} (batch={cfg["max_batch"]})')
        return 0
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    models = _select_models(args)
    logger.info(f'Models: {", ".join(models)}; formats: {", ".join(args.formats)}')
    results = [
        export_model(model_id, {**config, 'overwrite_config': args.overwrite_config}, args.formats)
        for model_id, config in models.items()
    ]
    print_summary(results)
    save_results_file(args.formats, models, results)
    if any_export_failed(results):
        logger.error('One or more exports failed -- see EXPORT SUMMARY above')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

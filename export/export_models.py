#!/usr/bin/env python3
"""
Unified YOLO Model Export Script
=================================

Export YOLO models in multiple formats for Triton deployment.

Formats:
--------
1. onnx         - Standard ONNX (no NMS, needs CPU post-processing)
2. trt          - Native TensorRT engine (no NMS, needs CPU post-processing)
3. onnx_end2end - ONNX with TensorRT EfficientNMS operators (GPU NMS)
4. trt_end2end  - TRT engine with compiled NMS (fastest, built from onnx_end2end)
5. all          - Export all formats (default)

Usage:
------
# Export all formats (default)
docker compose exec api python /app/export/export_models.py

# Export only end2end models
docker compose exec api python /app/export/export_models.py --formats onnx_end2end trt_end2end

# Export specific models
docker compose exec api python /app/export/export_models.py --models nano small

# Export everything
docker compose exec api python /app/export/export_models.py --formats all
"""

import os
import sys
from pathlib import Path


sys.path.insert(0, '/app/src')

# ----------------------------------------------------------------------------
# Toolchain guard — this exporter REQUIRES ultralytics < 8.4
# ----------------------------------------------------------------------------
# The EfficientNMS end2end patch (src/ultralytics_patches) targets the 8.3.x
# exporter API; ultralytics 8.4 (YOLO26) changed the end2end property and
# breaks it. The image ships a dedicated venv at /opt/venv-y11 with the pinned
# toolchain; when this script is launched under a newer ultralytics it
# transparently re-execs into that venv. YOLO26 exports (natively NMS-free,
# no patch needed) live in export/export_yolo26.py.
_Y11_VENV_PYTHON = os.environ.get('YOLO11_EXPORT_PYTHON', '/opt/venv-y11/bin/python')


def _ultralytics_is_pre_84() -> bool:
    try:
        from importlib.metadata import version

        major, minor, *_rest = version('ultralytics').split('.')
        return (int(major), int(minor)) < (8, 4)
    except Exception:
        return False


if not _ultralytics_is_pre_84():
    if Path(_Y11_VENV_PYTHON).exists() and os.environ.get('_Y11_REEXEC') != '1':
        os.environ['_Y11_REEXEC'] = '1'
        os.execv(_Y11_VENV_PYTHON, [_Y11_VENV_PYTHON, *sys.argv])  # nosec B606  re-exec of the pinned interpreter path constant with our own argv
    raise SystemExit(
        'export_models.py requires ultralytics<8.4 (EfficientNMS end2end patch) '
        f'and no pinned toolchain was found at {_Y11_VENV_PYTHON}. '
        'For YOLO26 models use export/export_yolo26.py instead.'
    )

import argparse  # noqa: E402  # imports follow the interpreter re-exec guard above
import json  # noqa: E402  # imports follow the interpreter re-exec guard above
import logging  # noqa: E402  # imports follow the interpreter re-exec guard above
from typing import Any  # noqa: E402  # imports follow the interpreter re-exec guard above

# Apply end2end patch for onnx_trt format
from ultralytics_patches import apply_end2end_patch  # noqa: E402  # imports follow the interpreter re-exec guard above


apply_end2end_patch()

# Import after patch (REQUIRED - patch must be applied before importing ultralytics)
from trt_utils import engine_output_dtypes  # noqa: E402  # imports follow the interpreter re-exec guard above
from ultralytics import YOLO  # noqa: E402  # imports follow the interpreter re-exec guard above
from yolo_export_settings import CONF_THRESHOLD, DEVICE, HALF, IMG_SIZE, IOU_THRESHOLD, WORKSPACE_GB  # noqa: E402  # imports follow the interpreter re-exec guard above
from yolo_formats import (  # noqa: E402  # imports follow the interpreter re-exec guard above
    check_gpu_memory,
    export_onnx_end2end,
    export_onnx_standard,
    export_trt_end2end,
    export_trt_standard,
)
from yolo_model_config import (  # noqa: E402  # imports follow the interpreter re-exec guard above
    extract_class_names,
    load_config_file,
    parse_custom_model,
    save_labels_file,
)
from yolo_triton_config import save_triton_config  # noqa: E402  # imports follow the interpreter re-exec guard above


# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Default configurations for standard YOLO11 model sizes
# These can be overridden via --config-file or extended via --custom-model
DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    'nano': {
        'pt_file': '/app/pytorch_models/yolo11n.pt',
        'triton_name': 'yolov11_nano',
        'max_batch': 128,  # Nano is small, A6000 can handle this
        'topk': 300,
    },
    'small': {
        'pt_file': '/app/pytorch_models/yolo11s.pt',
        'triton_name': 'yolov11_small',
        'max_batch': 64,
        'topk': 300,
    },
    'medium': {
        'pt_file': '/app/pytorch_models/yolo11m.pt',
        'triton_name': 'yolov11_medium',
        'max_batch': 32,
        'topk': 300,
    },
    'large': {
        'pt_file': '/app/pytorch_models/yolo11l.pt',
        'triton_name': 'yolov11_large',
        'max_batch': 16,  # Large model needs more memory
        'topk': 300,
    },
    'xlarge': {
        'pt_file': '/app/pytorch_models/yolo11x.pt',
        'triton_name': 'yolov11_xlarge',
        'max_batch': 8,  # XLarge is very memory intensive
        'topk': 300,
    },
}


# ============================================================================
# Helper Functions
# ============================================================================


def validate_pt_file(pt_file: str) -> bool:
    """Validate that PyTorch model file exists and is readable."""
    path = Path(pt_file)
    if not path.exists():
        logger.error(f'Model file not found: {pt_file}')
        return False
    if not path.is_file():
        logger.error(f'Model path is not a file: {pt_file}')
        return False
    if path.stat().st_size == 0:
        logger.error(f'Model file is empty: {pt_file}')
        return False
    return True


# ============================================================================
# Main Export Logic
# ============================================================================


def resolve_export_formats(formats: list[str], triton_name: str) -> list[str]:
    """Add ``onnx_end2end`` to ``formats`` when it's a missing dependency
    of a requested ``trt_end2end``.

    ``export_trt_end2end`` compiles the TensorRT engine from
    ``/app/models/{triton_name}_end2end/1/model.onnx``, which only
    ``onnx_end2end`` (or ``all``) produces. On a fresh clone, ``make
    export-models`` (and every other Make target that passes ``trt
    trt_end2end`` without ``onnx_end2end``) used to hit
    ``export_trt_end2end``'s "Run with --formats onnx_end2end first!"
    error every time, because nothing had built that ONNX yet.

    Leaves ``formats`` untouched when ``onnx_end2end``/``all`` is already
    requested, or when the ONNX from a prior run already exists on disk
    (no redundant re-export of an unrelated, already-built artifact).
    """
    if 'all' in formats or 'onnx_end2end' in formats or 'trt_end2end' not in formats:
        return list(formats)
    onnx_end2end_path = Path(f'/app/models/{triton_name}_end2end/1/model.onnx')
    if onnx_end2end_path.exists():
        return list(formats)
    return [*formats, 'onnx_end2end']


def export_model(
    model_id: str,
    config: dict[str, Any],
    formats: list[str],
    normalize_boxes: bool = True,
    generate_config: bool = False,
    save_labels: bool = False,
    overwrite_config: bool = False,
) -> dict[str, Any]:
    """
    Export a single model in specified formats.

    Args:
        model_id: Model identifier (nano, small, medium, large, xlarge, or custom)
        config: Model configuration dict
        formats: List of formats to export
        normalize_boxes: If True, output boxes in [0,1] range
        generate_config: If True, auto-generate Triton config.pbtxt files
        save_labels: If True, save class names to labels.txt
        overwrite_config: Replace an existing, different config.pbtxt instead of
            writing config.pbtxt.generated beside it

    Returns:
        Dict with export results for each format
    """
    logger.info('=' * 70)
    logger.info(f'Processing {model_id} -> {config["triton_name"]}')
    logger.info('=' * 70)

    pt_file = config['pt_file']

    # Validate input file
    if not validate_pt_file(pt_file):
        return {
            'model': model_id,
            'triton_name': config['triton_name'],
            'status': 'error',
            'error': f'Model file not found: {pt_file}',
        }

    resolved_formats = resolve_export_formats(formats, config['triton_name'])
    if resolved_formats != list(formats):
        logger.info(
            f'--formats requested trt_end2end without onnx_end2end and no cached '
            f'end2end ONNX was found; adding onnx_end2end to formats: {resolved_formats}'
        )
    formats = resolved_formats

    results = {
        'model': model_id,
        'triton_name': config['triton_name'],
        'max_batch': config['max_batch'],
        'topk': config['topk'],
    }

    # Load model once
    model = YOLO(pt_file)

    # Extract class information from model
    class_names = config.get('class_names') or extract_class_names(model)
    num_classes = config.get('num_classes') or len(class_names) or 80

    results['num_classes'] = num_classes
    results['class_names_count'] = len(class_names)

    logger.info(f'Model has {num_classes} classes')
    if class_names and len(class_names) <= 10:
        logger.info(f'Classes: {", ".join(class_names)}')
    elif class_names:
        logger.info(f'Classes: {", ".join(class_names[:5])}... (and {len(class_names) - 5} more)')

    # Export in requested formats
    if 'onnx' in formats or 'all' in formats:
        results['onnx'] = export_onnx_standard(model, config)

        # Save labels and generate config for ONNX model
        if results['onnx']['status'] == 'success':
            triton_name = config['triton_name']
            model_dir = Path(f'/app/models/{triton_name}')

            if save_labels and class_names:
                save_labels_file(class_names, model_dir)

            if generate_config:
                save_triton_config(
                    model_dir=model_dir,
                    triton_name=triton_name,
                    model_format='onnx',
                    max_batch=config['max_batch'],
                    num_classes=num_classes,
                    has_nms=False,
                    overwrite=overwrite_config,
                )

    if 'trt' in formats or 'all' in formats:
        # Reload model for clean state
        model = YOLO(pt_file)
        results['trt'] = export_trt_standard(model, config)

        # Save labels and generate config for TRT model
        if results['trt']['status'] == 'success':
            triton_name = config['triton_name']
            model_dir = Path(f'/app/models/{triton_name}_trt')

            if save_labels and class_names:
                save_labels_file(class_names, model_dir)

            if generate_config:
                save_triton_config(
                    model_dir=model_dir,
                    triton_name=f'{triton_name}_trt',
                    model_format='trt',
                    max_batch=config['max_batch'],
                    num_classes=num_classes,
                    has_nms=False,
                    overwrite=overwrite_config,
                )

    if 'onnx_end2end' in formats or 'all' in formats:
        # Reload model for clean state
        model = YOLO(pt_file)
        results['onnx_end2end'] = export_onnx_end2end(
            model, config, normalize_boxes=normalize_boxes
        )

        # Save labels for end2end ONNX model
        if results['onnx_end2end']['status'] == 'success':
            triton_name = config['triton_name']
            model_dir = Path(f'/app/models/{triton_name}_end2end')

            if save_labels and class_names:
                save_labels_file(class_names, model_dir)

    if 'trt_end2end' in formats or 'all' in formats:
        # This uses the end2end ONNX, doesn't need model reload
        results['trt_end2end'] = export_trt_end2end(config)

        # Save labels and generate config for TRT end2end model
        if results['trt_end2end']['status'] == 'success':
            triton_name = config['triton_name']
            model_dir = Path(f'/app/models/{triton_name}_trt_end2end')

            if save_labels and class_names:
                save_labels_file(class_names, model_dir)

            if generate_config:
                save_triton_config(
                    model_dir=model_dir,
                    triton_name=f'{triton_name}_trt_end2end',
                    model_format='trt_end2end',
                    max_batch=config['max_batch'],
                    num_classes=num_classes,
                    has_nms=True,
                    output_dtypes=engine_output_dtypes(model_dir / '1' / 'model.plan'),
                    overwrite=overwrite_config,
                )

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print export summary to logger."""
    logger.info('=' * 70)
    logger.info('EXPORT SUMMARY')
    logger.info('=' * 70)

    for result in results:
        logger.info(f'{result["model"]} ({result.get("triton_name", "?")}):')

        if result.get('status') == 'error':
            logger.info(f'  [FAIL] {result.get("error", "unknown error")}')
            continue

        if 'onnx' in result:
            status = '[OK]' if result['onnx']['status'] == 'success' else '[FAIL]'
            logger.info(f'  {status} ONNX Standard: {result["onnx"]["status"]}')
            if result['onnx']['status'] == 'success':
                logger.info(f'       Host: {result["onnx"]["host_path"]}')

        if 'trt' in result:
            status = '[OK]' if result['trt']['status'] == 'success' else '[FAIL]'
            logger.info(f'  {status} TRT Standard: {result["trt"]["status"]}')
            if result['trt']['status'] == 'success':
                logger.info(f'       Host: {result["trt"]["host_path"]}')

        if 'onnx_end2end' in result:
            status = '[OK]' if result['onnx_end2end']['status'] == 'success' else '[FAIL]'
            logger.info(f'  {status} ONNX End2End: {result["onnx_end2end"]["status"]}')
            if result['onnx_end2end']['status'] == 'success':
                logger.info(f'       Host: {result["onnx_end2end"]["host_path"]}')
                logger.info('       NMS: GPU (TRT operators)')

        if 'trt_end2end' in result:
            status = '[OK]' if result['trt_end2end']['status'] == 'success' else '[FAIL]'
            logger.info(f'  {status} TRT End2End: {result["trt_end2end"]["status"]}')
            if result['trt_end2end']['status'] == 'success':
                logger.info(f'       Host: {result["trt_end2end"]["host_path"]}')
                logger.info('       NMS: Compiled (maximum performance!)')

    logger.info('-' * 70)
    logger.info('FORMAT COMPARISON')
    logger.info('-' * 70)
    logger.info('1. ONNX Standard: [84, 8400] -> CPU NMS, baseline speed')
    logger.info('2. TRT Standard:  [84, 8400] -> CPU NMS, 1.5x faster')
    logger.info('3. ONNX End2End:  NMS outputs -> GPU NMS via TRT EP, 2-3x faster')
    logger.info('4. TRT End2End:   NMS outputs -> compiled GPU NMS, 3-5x faster')
    logger.info('=' * 70)


def _any_export_failed(results: list[dict[str, Any]]) -> bool:
    """Whether any model/format in ``results`` reports an error.

    ``main()`` used to always exit 0 regardless of what ``results``
    said -- a total, silent failure (e.g. every format's TensorRT build
    rejected by a poisoned CUDA_VISIBLE_DEVICES, see setup_trt_builder)
    looked identical to a clean run to anything checking the exit code
    (CI, ``make``, an operator's shell).
    """
    for result in results:
        if result.get('status') == 'error':
            return True
        for value in result.values():
            if isinstance(value, dict) and value.get('status') == 'error':
                return True
    return False


def main() -> int:
    """Main entry point for export script. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description='Export YOLO models in multiple formats for Triton',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all formats for default models
  python export_models.py

  # Export only end2end models
  python export_models.py --formats onnx_end2end trt_end2end

  # Export specific built-in models (nano, small, medium, large, xlarge)
  python export_models.py --models nano small large

  # Export a custom trained model
  python export_models.py --custom-model /path/to/my_model.pt

  # Export custom model with specific name and batch size
  python export_models.py --custom-model /path/to/my_model.pt:custom_name:64

  # Export using a YAML configuration file
  python export_models.py --config-file /path/to/models.yaml

  # Generate Triton configs and labels for exported models
  python export_models.py --models small --generate-config --save-labels
        """,
    )

    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'trt', 'onnx_end2end', 'trt_end2end', 'all'],
        default=['all'],
        help='Formats to export (default: all)',
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help=(
            'Built-in models to export: nano, small, medium, large, xlarge. '
            'If not specified and no --custom-model or --config-file, exports small only.'
        ),
    )

    parser.add_argument(
        '--custom-model',
        nargs='+',
        dest='custom_models',
        metavar='PATH[:NAME][:BATCH]',
        help=(
            'Custom model(s) to export. Format: path/to/model.pt[:triton_name][:max_batch]. '
            'Examples: my_model.pt, my_model.pt:custom_name, my_model.pt:custom_name:64'
        ),
    )

    parser.add_argument(
        '--config-file',
        dest='config_file',
        metavar='PATH',
        help='YAML configuration file with model definitions (overrides --models)',
    )

    parser.add_argument(
        '--normalize-boxes',
        action='store_true',
        default=False,
        help='Output boxes in [0, 1] normalized range instead of pixel coords (recommended)',
    )

    parser.add_argument(
        '--generate-config',
        action='store_true',
        default=False,
        help='Auto-generate Triton config.pbtxt files for exported models',
    )

    parser.add_argument(
        '--overwrite-config',
        action='store_true',
        default=False,
        help='With --generate-config: replace an existing config.pbtxt that differs '
        '(default: write config.pbtxt.generated beside it)',
    )

    parser.add_argument(
        '--save-labels',
        action='store_true',
        default=False,
        help='Save class names to labels.txt in each model directory',
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available built-in models and exit',
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging',
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print('Available built-in YOLO11 models:')
        print('-' * 50)
        for model_id, cfg in DEFAULT_MODELS.items():
            print(f'  {model_id:10} -> {cfg["triton_name"]:25} (batch={cfg["max_batch"]})')
        print('-' * 50)
        print('\nUsage: --models nano small medium large xlarge')
        print('Custom: --custom-model /path/to/model.pt[:name][:batch]')
        sys.exit(0)

    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build model configurations from various sources
    models_to_export: dict[str, dict[str, Any]] = {}

    # 1. Load from config file if specified
    if args.config_file:
        try:
            models_to_export.update(load_config_file(args.config_file))
        except (FileNotFoundError, ValueError) as e:
            logger.error(f'Config file error: {e}')
            sys.exit(1)

    # 2. Add built-in models if specified
    if args.models:
        for model_id in args.models:
            if model_id in DEFAULT_MODELS:
                models_to_export[model_id] = DEFAULT_MODELS[model_id].copy()
            else:
                logger.error(
                    f"Unknown model '{model_id}'. "
                    f'Available: {", ".join(DEFAULT_MODELS.keys())}. '
                    f'Use --custom-model for custom models.'
                )
                sys.exit(1)

    # 3. Add custom models if specified
    if args.custom_models:
        for custom_arg in args.custom_models:
            try:
                model_id, model_config = parse_custom_model(custom_arg)
                models_to_export[model_id] = model_config
            except (FileNotFoundError, ValueError) as e:
                logger.error(f'Custom model error: {e}')
                sys.exit(1)

    # 4. Default: export 'small' if nothing specified
    if not models_to_export:
        logger.info('No models specified, defaulting to small')
        models_to_export['small'] = DEFAULT_MODELS['small'].copy()

    logger.info('=' * 70)
    logger.info('YOLO11 Unified Export Script')
    logger.info('=' * 70)
    logger.info(f'Models: {", ".join(models_to_export.keys())}')
    logger.info(f'Formats: {", ".join(args.formats)}')
    logger.info(f'Image size: {IMG_SIZE}x{IMG_SIZE}')
    logger.info(f'Device: {f"cuda:{DEVICE}" if DEVICE != "cpu" else "cpu"}')
    logger.info(f'Precision: {"FP16" if HALF else "FP32"}')
    logger.info(f'Workspace: {WORKSPACE_GB}GB')
    if 'onnx_end2end' in args.formats or 'trt_end2end' in args.formats or 'all' in args.formats:
        logger.info(f'NMS IoU threshold: {IOU_THRESHOLD}')
        logger.info(f'Confidence threshold: {CONF_THRESHOLD}')
        box_mode = '(recommended)' if args.normalize_boxes else '(pixel coords 640x640)'
        logger.info(f'Normalize boxes: {args.normalize_boxes} {box_mode}')
    if args.generate_config:
        logger.info('Auto-generate Triton configs: enabled')
    if args.save_labels:
        logger.info('Save labels.txt: enabled')
    logger.info('=' * 70)

    # Check GPU availability
    if not check_gpu_memory():
        logger.error('GPU check failed. Aborting.')
        sys.exit(1)

    results = []

    for model_id, model_config in models_to_export.items():
        result = export_model(
            model_id,
            model_config,
            args.formats,
            normalize_boxes=args.normalize_boxes,
            generate_config=args.generate_config,
            save_labels=args.save_labels,
            overwrite_config=args.overwrite_config,
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Save results
    results_file = Path('/app/scripts/export_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with Path(results_file).open('w') as f:
        json.dump(
            {
                'formats': args.formats,
                'models': list(models_to_export.keys()),
                'results': results,
            },
            f,
            indent=2,
        )
        f.write('\n')
    logger.info(f'Results saved to: {results_file}')

    # Show current model structure
    logger.info('=' * 70)
    logger.info('CURRENT MODEL REPOSITORY')
    logger.info('=' * 70)
    models_dir = Path('/app/models')
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir() and (
                model_dir.name.startswith('yolov11') or model_dir.name.startswith('yolo11')
            ):
                has_model = (model_dir / '1' / 'model.onnx').exists() or (
                    model_dir / '1' / 'model.plan'
                ).exists()
                has_config = (model_dir / 'config.pbtxt').exists()
                has_labels = (model_dir / 'labels.txt').exists()
                status = '[OK]' if has_model else '[--]'
                config_status = '[OK]' if has_config else '[!!]'
                labels_status = '[OK]' if has_labels else '[--]'
                logger.info(
                    f'{status} {model_dir.name:35} config: {config_status} labels: {labels_status}'
                )
    logger.info('=' * 70)

    if _any_export_failed(results):
        logger.error('One or more exports failed -- see EXPORT SUMMARY above')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

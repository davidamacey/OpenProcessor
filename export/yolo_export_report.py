"""Reporting for the YOLO11 exporter: the export summary, failure check, results file and repository listing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


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


def save_results_file(
    formats: list[str], models_to_export: dict[str, dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    """Write the machine-readable export results next to the other generated scripts output."""
    results_file = Path('/app/scripts/export_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with Path(results_file).open('w') as f:
        json.dump(
            {
                'formats': formats,
                'models': list(models_to_export.keys()),
                'results': results,
            },
            f,
            indent=2,
        )
        f.write('\n')
    logger.info(f'Results saved to: {results_file}')


def log_model_repository() -> None:
    """Log which YOLO11 model directories are present, with config and labels status."""
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

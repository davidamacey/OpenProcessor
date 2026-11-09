"""Reporting of the Ultralytics detector export: the summary, the failure check, the results file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

RESULTS_FILE = Path('/app/scripts/export_results.json')


def print_summary(results: list[dict[str, Any]]) -> None:
    """Log one block per model: its status, the ONNX and the engine it produced."""
    logger.info('=' * 70)
    logger.info('EXPORT SUMMARY')
    logger.info('=' * 70)
    for result in results:
        logger.info(f'{result["model"]} ({result.get("triton_name", "?")}):')
        if result.get('status') != 'success':
            logger.info(f'  [FAIL] {result.get("error", "unknown error")}')
            continue
        logger.info(f'  [OK]   ONNX: {result["onnx"]}')
        if 'trt' in result:
            logger.info(f'  [OK]   TensorRT engine: {result["trt"]}')
    logger.info('=' * 70)


def any_export_failed(results: list[dict[str, Any]]) -> bool:
    """Whether any model did not export: the exit code of the script follows it."""
    return any(result.get('status') != 'success' for result in results)


def save_results_file(
    formats: list[str], models_to_export: dict[str, dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    """Write the machine-readable export results next to the other generated scripts output."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open('w') as f:
        json.dump(
            {'formats': formats, 'models': list(models_to_export), 'results': results},
            f,
            indent=2,
        )
        f.write('\n')
    logger.info(f'Results saved to: {RESULTS_FILE}')

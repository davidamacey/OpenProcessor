"""Tests for the F-08 crash in ``export/export_models.py``.

A fresh clone has no ``make`` target that fetches ``yolo11s.pt`` before
``make export-models`` runs (only ``scripts/setup.sh`` does, which the
runbook tells operators to skip on a shared host). The resulting
``FileNotFoundError``-style failure used to be masked entirely:
``export_model`` returned an error dict without a ``triton_name`` key, and
``print_summary`` unconditionally read ``result["triton_name"]`` before
checking status, so the *real* "model file not found" message never made it
to the operator — they saw a bare ``KeyError`` traceback instead.

``export_models.py`` imports ``tensorrt`` and ``ultralytics_patches`` at
module level (it's a script meant to run inside the ``yolo-api`` container,
which has both), so these tests skip cleanly wherever that stack isn't
installed — same convention as ``pytest.importorskip('onnxruntime')`` in
``tests/test_pe_image_encoder_export.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


pytest.importorskip('tensorrt')

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import export_models  # noqa: E402


class TestMissingPtFileSurfacesARealError:
    def test_export_model_error_dict_carries_triton_name(self, tmp_path: Path) -> None:
        config = {
            'pt_file': str(tmp_path / 'does_not_exist.pt'),
            'triton_name': 'yolov11_small',
            'max_batch': 16,
            'topk': 100,
        }
        result = export_models.export_model('small', config, formats=['trt'])
        assert result['status'] == 'error'
        assert result['triton_name'] == 'yolov11_small'
        assert 'does_not_exist.pt' in result['error']

    def test_print_summary_does_not_raise_on_an_error_result(self, caplog) -> None:
        caplog.set_level(logging.INFO)
        results = [
            {
                'model': 'small',
                'triton_name': 'yolov11_small',
                'status': 'error',
                'error': 'Model file not found: /app/pytorch_models/yolo11s.pt',
            }
        ]
        # This is the exact crash from the finding: print_summary used to
        # raise KeyError('triton_name') here instead of printing the error.
        export_models.print_summary(results)
        assert 'Model file not found' in caplog.text

    def test_print_summary_still_handles_a_successful_result(self, caplog) -> None:
        caplog.set_level(logging.INFO)
        results = [
            {
                'model': 'small',
                'triton_name': 'yolov11_small',
                'trt': {'status': 'success', 'host_path': '/app/models/yolov11_small_trt'},
            }
        ]
        export_models.print_summary(results)
        assert 'yolov11_small' in caplog.text

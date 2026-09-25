"""Tests for the F-08 crash and F-12 fail-open exit code in
``export/export_models.py``.

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
from typing import Any

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


class TestAnyExportFailed:
    """F-12(c): main() used to exit 0 no matter what results said -- a
    total, silent failure looked identical to a clean run to CI/make/an
    operator's shell."""

    def test_a_top_level_error_result_is_a_failure(self) -> None:
        results: list[dict[str, Any]] = [{'model': 'small', 'status': 'error', 'error': 'boom'}]
        assert export_models._any_export_failed(results) is True

    def test_a_per_format_error_is_a_failure(self) -> None:
        """The realistic F-12 shape: export_model() succeeds overall but
        an individual format (e.g. 'trt') reports its own error dict."""
        results: list[dict[str, Any]] = [
            {
                'model': 'small',
                'triton_name': 'yolov11_small',
                'trt': {'status': 'error', 'error': 'Failed to build engine'},
                'trt_end2end': {'status': 'error', 'error': 'Failed to build engine'},
            }
        ]
        assert export_models._any_export_failed(results) is True

    def test_all_successful_formats_is_not_a_failure(self) -> None:
        results: list[dict[str, Any]] = [
            {
                'model': 'small',
                'triton_name': 'yolov11_small',
                'trt': {'status': 'success', 'host_path': '/app/models/yolov11_small_trt'},
            }
        ]
        assert export_models._any_export_failed(results) is False

    def test_empty_results_is_not_a_failure(self) -> None:
        assert export_models._any_export_failed([]) is False


class TestCudaVisibleDevicesRepairBeforeTrtBuild:
    """F-09/F-12(a): /opt/venv-y11's CPU-only torch means the ONNX-export
    step always runs Ultralytics with device='cpu', which sets
    CUDA_VISIBLE_DEVICES=-1 as a side effect -- in the same process, this
    then blinds the TensorRT builder created right after it
    ("CUDA initialization failure with error: 100"), even though the
    engine build doesn't use torch's CUDA at all.

    ``setup_trt_builder`` still constructs a real ``trt.Builder`` (a real
    CUDA context) after the env-var repair, which needs an actual GPU --
    not something a unit test should require. ``export_models.trt`` is
    patched with a bare stand-in so only the repair logic itself is
    under test here.
    """

    @pytest.fixture(autouse=True)
    def _fake_trt(self, monkeypatch: pytest.MonkeyPatch):
        from unittest.mock import MagicMock

        fake_trt = MagicMock()
        fake_trt.Logger.INFO = 0
        fake_config = MagicMock()
        fake_trt.Builder.return_value.create_builder_config.return_value = fake_config
        monkeypatch.setattr(export_models, 'trt', fake_trt)
        return fake_trt

    def test_poisoned_cuda_visible_devices_is_cleared_before_building(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
        export_models.setup_trt_builder()
        assert 'CUDA_VISIBLE_DEVICES' not in export_models.os.environ

    def test_an_unset_cuda_visible_devices_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
        export_models.setup_trt_builder()
        assert 'CUDA_VISIBLE_DEVICES' not in export_models.os.environ

    def test_a_legitimate_gpu_pin_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the exact '-1' (Ultralytics' CPU-mode sentinel) is
        cleared -- an operator's own GPU selection must survive."""
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
        export_models.setup_trt_builder()
        assert export_models.os.environ['CUDA_VISIBLE_DEVICES'] == '0'


class TestSaveTritonConfigWriteOnlyOnChange:
    """F-15: models/*/config.pbtxt is tracked in git. Every export run
    called save_triton_config unconditionally, rewriting the file (new
    mtime, often byte-identical content) and dirtying the tree even when
    nothing about the generated config actually changed."""

    def test_does_not_rewrite_an_identical_config(self, tmp_path: Path) -> None:
        model_dir = tmp_path / 'yolov11_small_trt_end2end'
        model_dir.mkdir()
        path1 = export_models.save_triton_config(
            model_dir, 'yolov11_small_trt_end2end', 'trt_end2end', max_batch=64, has_nms=True
        )
        mtime_before = path1.stat().st_mtime_ns
        content_before = path1.read_text()

        path2 = export_models.save_triton_config(
            model_dir, 'yolov11_small_trt_end2end', 'trt_end2end', max_batch=64, has_nms=True
        )

        assert path2 == path1
        assert path2.read_text() == content_before
        assert path2.stat().st_mtime_ns == mtime_before, 'identical config must not be rewritten'

    def test_does_rewrite_when_content_actually_changes(self, tmp_path: Path) -> None:
        model_dir = tmp_path / 'yolov11_small_trt_end2end'
        model_dir.mkdir()
        export_models.save_triton_config(
            model_dir, 'yolov11_small_trt_end2end', 'trt_end2end', max_batch=64, has_nms=True
        )
        path = export_models.save_triton_config(
            model_dir, 'yolov11_small_trt_end2end', 'trt_end2end', max_batch=32, has_nms=True
        )
        assert 'max_batch_size: 32' in path.read_text()

"""F-42 (fresh-start E2E findings 2026-09-25, round 2): class-name
resolution must be scoped to the model that actually produced the
detection, not a hardcoded assumption that every caller is the stock
COCO detector.

Covers three layers: the pure formatter
(src.utils.affine.format_detections_from_triton), the Triton client
wrapper (src.clients.triton_client.TritonClient.format_detections), and
the service layer that threads model_name through from the /detect and
/detect/batch routes (src.services.inference.InferenceService).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.clients.triton_client import TritonClient
from src.utils import class_names as cn
from src.utils.affine import format_detections_from_triton


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch: pytest.MonkeyPatch):
    cn.clear_class_name_cache()
    monkeypatch.delenv('OP_TRITON_MODEL_REPO', raising=False)
    yield
    cn.clear_class_name_cache()


def _fake_result(class_ids: list[int]) -> dict:
    n = len(class_ids)
    return {
        'boxes': np.tile(np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32), (n, 1)),
        'scores': np.full(n, 0.9, dtype=np.float32),
        'classes': np.array(class_ids, dtype=np.float32),
    }


class TestFormatDetectionsFromTriton:
    def test_model_name_none_keeps_historical_coco_only_behavior(self) -> None:
        out = format_detections_from_triton(_fake_result([0, 2]), model_name=None)
        assert out[0]['class_name'] == 'person'
        assert out[1]['class_name'] == 'car'

    def test_model_name_resolves_from_that_models_own_labels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
        model_dir = tmp_path / 'vehicle_detector_v1'
        model_dir.mkdir()
        (model_dir / 'labels.txt').write_text('car\ntruck\nbus\n')

        out = format_detections_from_triton(_fake_result([0, 1]), model_name='vehicle_detector_v1')

        assert out[0]['class_name'] == 'car'
        assert out[1]['class_name'] == 'truck'

    def test_a_promoted_models_class_ids_never_borrow_coco_words(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact regression: F-42's bus/train/truck mislabeling."""
        monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
        model_dir = tmp_path / 'vehicle_detector_v1'
        model_dir.mkdir()
        # class 5 is 'bus' in COCO -- this model's class 5 is something else.
        (model_dir / 'labels.txt').write_text('\n'.join(['a', 'b', 'c', 'd', 'e', 'widget']))

        out = format_detections_from_triton(_fake_result([5]), model_name='vehicle_detector_v1')

        assert out[0]['class_name'] == 'widget'
        assert out[0]['class_name'] != 'bus'


class TestTritonClientFormatDetections:
    def test_forwards_model_name_to_the_resolver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
        model_dir = tmp_path / 'promoted_v2'
        model_dir.mkdir()
        (model_dir / 'labels.txt').write_text('widget\n')

        out = TritonClient.format_detections(_fake_result([0]), model_name='promoted_v2')

        assert out[0]['class_name'] == 'widget'

    def test_default_model_name_none_is_backward_compatible(self) -> None:
        out = TritonClient.format_detections(_fake_result([2]))
        assert out[0]['class_name'] == 'car'


class TestInferenceServiceThreadsModelNameThrough:
    """The actual F-42 symptom lived here: /detect?model_name=<promoted>
    called format_detections() with no model_name at all."""

    def test_detect_passes_its_model_name_to_format_detections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.services import inference as inference_module

        fake_client = MagicMock()
        fake_client.infer_yolo_end2end.return_value = {'boxes': [], 'scores': [], 'classes': []}
        fake_client.format_detections.return_value = []
        monkeypatch.setattr(inference_module, 'get_triton_client', lambda *_a, **_k: fake_client)
        monkeypatch.setattr(
            inference_module, 'decode_image', lambda *_a, **_k: np.zeros((10, 10, 3), np.uint8)
        )
        monkeypatch.setattr(inference_module, 'validate_image', lambda *_a, **_k: None)

        service = inference_module.InferenceService()
        service.detect(b'fake-jpeg-bytes', model_name='vehicle_detector_v1')

        fake_client.format_detections.assert_called_once()
        _args, kwargs = fake_client.format_detections.call_args
        assert kwargs.get('model_name') == 'vehicle_detector_v1'

    def test_detect_batch_passes_its_model_name_to_format_detections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.services import inference as inference_module

        fake_client = MagicMock()
        fake_client.infer_yolo_end2end_batch.return_value = [
            {'boxes': [], 'scores': [], 'classes': []}
        ]
        fake_client.format_detections.return_value = []
        monkeypatch.setattr(inference_module, 'get_triton_client', lambda *_a, **_k: fake_client)
        monkeypatch.setattr(
            inference_module, 'decode_image', lambda *_a, **_k: np.zeros((10, 10, 3), np.uint8)
        )
        monkeypatch.setattr(inference_module, 'validate_image', lambda *_a, **_k: None)

        service = inference_module.InferenceService()
        service.detect_batch([b'fake-jpeg-bytes'], model_name='vehicle_detector_v1')

        fake_client.format_detections.assert_called_once()
        _args, kwargs = fake_client.format_detections.call_args
        assert kwargs.get('model_name') == 'vehicle_detector_v1'

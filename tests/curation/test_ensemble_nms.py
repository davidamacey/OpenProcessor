"""Tests for src.services.detection.ensemble_nms.

The module sys.path-injects a vendored external YOLOv5 fork at import
time and raises RuntimeError if it isn't found (see the module
docstring and the allowlist note in tests/test_dependency_manifest.py).
No such fork is vendored in this repo, so this test points
DETECTION_YOLOV5_FORK at a small local fixture (tests/fixtures/yolov5_fork)
providing a plain, correct NMS implementation — not a copy of any
reference/proprietary fork — before importing the module under test.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


_FIXTURE_FORK = Path(__file__).resolve().parent.parent / 'fixtures' / 'yolov5_fork'
os.environ.setdefault('DETECTION_YOLOV5_FORK', str(_FIXTURE_FORK))

from src.services.detection.ensemble_nms import apply_ensemble_nms  # noqa: E402


def _make_raw(dets: list[tuple[float, float, float, float, float, int]], nc: int = 2) -> np.ndarray:
    """Build a ``[1, N, 5+nc]`` raw ensemble-detector output.

    Each det is ``(cx, cy, w, h, obj_conf, class_id)``; the class score
    row is one-hot at 1.0 for ``class_id`` so ``obj_conf * cls_conf ==
    obj_conf``, keeping the fixture easy to reason about.
    """
    rows = []
    for cx, cy, w, h, obj_conf, cls in dets:
        row = [cx, cy, w, h, obj_conf] + [0.0] * nc
        row[5 + cls] = 1.0
        rows.append(row)
    return np.array([rows], dtype=np.float32)


class TestEmptyInput:
    def test_empty_batch_returns_empty_list(self) -> None:
        raw = np.zeros((1, 0, 7), dtype=np.float32)
        out = apply_ensemble_nms(raw)
        assert out == [[]]

    def test_all_below_confidence_floor_returns_empty(self) -> None:
        raw = _make_raw([(50, 50, 20, 20, 0.05, 0)])
        out = apply_ensemble_nms(raw, conf_thres=0.25)
        assert out == [[]]


class TestSuppressionAtIouBoundary:
    def test_high_overlap_suppresses_lower_score(self) -> None:
        # Two near-identical boxes, same class -> only the higher-score one survives.
        raw = _make_raw(
            [
                (50, 50, 20, 20, 0.9, 0),
                (51, 51, 20, 20, 0.6, 0),
            ]
        )
        out = apply_ensemble_nms(raw, conf_thres=0.25, iou_thres=0.45)
        assert len(out[0]) == 1
        assert out[0][0]['score'] == pytest.approx(0.9)

    def test_low_overlap_keeps_both(self) -> None:
        raw = _make_raw(
            [
                (10, 10, 10, 10, 0.9, 0),
                (100, 100, 10, 10, 0.6, 0),
            ]
        )
        out = apply_ensemble_nms(raw, conf_thres=0.25, iou_thres=0.45)
        assert len(out[0]) == 2


class TestClassAwareness:
    def test_different_classes_both_survive_despite_overlap(self) -> None:
        # Identical boxes but different classes -> class-aware NMS keeps both.
        raw = _make_raw(
            [
                (50, 50, 20, 20, 0.9, 0),
                (50, 50, 20, 20, 0.8, 1),
            ]
        )
        out = apply_ensemble_nms(raw, conf_thres=0.25, iou_thres=0.45, agnostic=False)
        classes = sorted(d['class_id'] for d in out[0])
        assert classes == [0, 1]

    def test_agnostic_mode_suppresses_across_classes(self) -> None:
        raw = _make_raw(
            [
                (50, 50, 20, 20, 0.9, 0),
                (50, 50, 20, 20, 0.8, 1),
            ]
        )
        out = apply_ensemble_nms(raw, conf_thres=0.25, iou_thres=0.45, agnostic=True)
        assert len(out[0]) == 1
        assert out[0][0]['class_id'] == 0

    def test_classes_filter_drops_unwanted_class(self) -> None:
        raw = _make_raw(
            [
                (10, 10, 10, 10, 0.9, 0),
                (100, 100, 10, 10, 0.8, 1),
            ]
        )
        out = apply_ensemble_nms(raw, conf_thres=0.25, classes=[0])
        assert len(out[0]) == 1
        assert out[0][0]['class_id'] == 0


class TestOutputShape:
    def test_box_is_x1y1x2y2_in_pixel_space(self) -> None:
        raw = _make_raw([(50, 50, 20, 20, 0.9, 0)])
        out = apply_ensemble_nms(raw, conf_thres=0.25)
        [det] = out[0]
        x1, y1, x2, y2 = det['box']
        assert x1 == pytest.approx(40.0)
        assert y1 == pytest.approx(40.0)
        assert x2 == pytest.approx(60.0)
        assert y2 == pytest.approx(60.0)

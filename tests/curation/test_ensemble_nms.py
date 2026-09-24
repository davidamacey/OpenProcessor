"""Tests for src.services.detection.ensemble_nms (native YOLOv5-semantics NMS)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from src.services.detection.ensemble_nms import apply_ensemble_nms, non_max_suppression


if TYPE_CHECKING:
    from collections.abc import Sequence


def _make_raw(
    dets: Sequence[tuple[float, float, float, float, float, int]], nc: int = 2
) -> np.ndarray:
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

    def test_malformed_shape_raises(self) -> None:
        with pytest.raises(ValueError, match='5 \\+ nc'):
            apply_ensemble_nms(np.zeros((4, 7), dtype=np.float32))


class TestConfidence:
    def test_objectness_gates_before_class_scores(self) -> None:
        # With sigmoid class scores the gate is implied by the product check;
        # it is only observable when a class score exceeds 1. obj 0.2 fails
        # the 0.25 gate even though obj * cls = 0.4 would pass.
        raw = np.array([[[50, 50, 20, 20, 0.2, 2.0, 0.0]]], dtype=np.float32)
        assert apply_ensemble_nms(raw, conf_thres=0.25) == [[]]

    def test_objectness_gate_uses_obj_not_product(self) -> None:
        # obj 0.3 passes the objectness gate at 0.1; product 0.3 * 0.5 = 0.15
        # also passes 0.1. At conf 0.2 the objectness gate passes but the
        # product (0.15) does not.
        raw = np.array([[[50, 50, 20, 20, 0.3, 0.5, 0.0]]], dtype=np.float32)
        [[det]] = apply_ensemble_nms(raw, conf_thres=0.1)
        assert det['score'] == pytest.approx(0.15)
        assert apply_ensemble_nms(raw, conf_thres=0.2) == [[]]

    def test_class_score_times_objectness_picks_best_class(self) -> None:
        raw = np.array([[[50, 50, 20, 20, 0.8, 0.3, 0.9, 0.5]]], dtype=np.float32)
        [[det]] = apply_ensemble_nms(raw, conf_thres=0.25)
        assert det['class_id'] == 1
        assert det['score'] == pytest.approx(0.8 * 0.9)

    def test_threshold_is_strict(self) -> None:
        # Exactly-at-threshold is rejected (``>``, not ``>=``); 0.5 is exact in float32.
        raw = _make_raw([(50, 50, 20, 20, 0.5, 0)])
        assert apply_ensemble_nms(raw, conf_thres=0.5) == [[]]
        assert len(apply_ensemble_nms(raw, conf_thres=0.4999)[0]) == 1

    def test_out_of_range_thresholds_raise(self) -> None:
        raw = _make_raw([(50, 50, 20, 20, 0.9, 0)])
        with pytest.raises(ValueError, match='conf_thres'):
            apply_ensemble_nms(raw, conf_thres=1.5)
        with pytest.raises(ValueError, match='iou_thres'):
            apply_ensemble_nms(raw, iou_thres=-0.1)


class TestSuppressionAtIouBoundary:
    def test_high_overlap_suppresses_lower_score(self) -> None:
        raw = _make_raw([(50, 50, 20, 20, 0.9, 0), (51, 51, 20, 20, 0.6, 0)])
        out = apply_ensemble_nms(raw, conf_thres=0.25, iou_thres=0.45)
        assert len(out[0]) == 1
        assert out[0][0]['score'] == pytest.approx(0.9)

    def test_low_overlap_keeps_both(self) -> None:
        raw = _make_raw([(10, 10, 10, 10, 0.9, 0), (100, 100, 10, 10, 0.6, 0)])
        out = apply_ensemble_nms(raw, conf_thres=0.25, iou_thres=0.45)
        assert len(out[0]) == 2

    def test_iou_exactly_at_threshold_keeps_both(self) -> None:
        # Two 10x10 boxes offset by 5 px on x: inter 50, union 150 -> IoU 1/3.
        # torchvision suppresses only IoU > thres.
        raw = _make_raw([(10, 10, 10, 10, 0.9, 0), (15, 10, 10, 10, 0.8, 0)])
        assert len(apply_ensemble_nms(raw, iou_thres=0.34)[0]) == 2
        assert len(apply_ensemble_nms(raw, iou_thres=0.33)[0]) == 1


class TestClassAwareness:
    def test_different_classes_both_survive_despite_overlap(self) -> None:
        raw = _make_raw([(50, 50, 20, 20, 0.9, 0), (50, 50, 20, 20, 0.8, 1)])
        out = apply_ensemble_nms(raw, conf_thres=0.25, iou_thres=0.45, agnostic=False)
        assert sorted(d['class_id'] for d in out[0]) == [0, 1]

    def test_agnostic_mode_suppresses_across_classes(self) -> None:
        raw = _make_raw([(50, 50, 20, 20, 0.9, 0), (50, 50, 20, 20, 0.8, 1)])
        out = apply_ensemble_nms(raw, conf_thres=0.25, iou_thres=0.45, agnostic=True)
        assert len(out[0]) == 1
        assert out[0][0]['class_id'] == 0

    def test_class_offset_holds_for_large_input_boxes(self) -> None:
        # Boxes near the 1280-px input edge, different classes, identical
        # geometry: the per-class offset must still separate them.
        raw = _make_raw([(1200, 1200, 150, 150, 0.9, 0), (1200, 1200, 150, 150, 0.8, 1)])
        assert len(apply_ensemble_nms(raw)[0]) == 2

    def test_classes_filter_drops_unwanted_class(self) -> None:
        raw = _make_raw([(10, 10, 10, 10, 0.9, 0), (100, 100, 10, 10, 0.8, 1)])
        out = apply_ensemble_nms(raw, conf_thres=0.25, classes=[0])
        assert len(out[0]) == 1
        assert out[0][0]['class_id'] == 0

    def test_classes_filter_applies_before_nms(self) -> None:
        # The class-1 box would not suppress the class-0 box anyway (class
        # aware); in agnostic mode it would, unless filtered out first.
        raw = _make_raw([(50, 50, 20, 20, 0.9, 1), (50, 50, 20, 20, 0.8, 0)])
        out = apply_ensemble_nms(raw, classes=[0], agnostic=True)
        assert [d['class_id'] for d in out[0]] == [0]


class TestCaps:
    def test_max_det_caps_output_keeping_highest_scores(self) -> None:
        dets = [(20.0 * i + 10, 10, 8, 8, 0.3 + 0.01 * i, 0) for i in range(40)]
        out = apply_ensemble_nms(_make_raw(dets), max_det=5)
        assert len(out[0]) == 5
        scores = [d['score'] for d in out[0]]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(0.69)

    def test_max_nms_caps_candidates_before_suppression(self) -> None:
        dets = [(20.0 * i + 10, 10, 8, 8, 0.3 + 0.01 * i, 0) for i in range(10)]
        pred = torch.from_numpy(_make_raw(dets))
        [det] = non_max_suppression(pred, max_nms=3)
        assert det.shape == (3, 6)
        assert det[:, 4].tolist() == pytest.approx([0.39, 0.38, 0.37])


class TestOutputShape:
    def test_box_is_x1y1x2y2_in_pixel_space(self) -> None:
        raw = _make_raw([(50, 50, 20, 30, 0.9, 0)])
        [det] = apply_ensemble_nms(raw, conf_thres=0.25)[0]
        assert det['box'] == pytest.approx([40.0, 35.0, 60.0, 65.0])
        assert set(det) == {'box', 'score', 'class_id'}
        assert isinstance(det['score'], float)
        assert isinstance(det['class_id'], int)

    def test_batch_of_images_is_processed_independently(self) -> None:
        a = _make_raw([(50, 50, 20, 20, 0.9, 0), (51, 51, 20, 20, 0.8, 0)])
        b = _make_raw([(10, 10, 10, 10, 0.1, 1), (10, 10, 10, 10, 0.1, 1)])
        c = _make_raw([(10, 10, 10, 10, 0.7, 1), (200, 200, 10, 10, 0.6, 0)])
        out = apply_ensemble_nms(np.concatenate([a, b, c]))
        assert [len(o) for o in out] == [1, 0, 2]
        assert out[2][0]['class_id'] == 1

    def test_numpy_and_torch_inputs_agree(self) -> None:
        raw = _make_raw(
            [(50, 50, 20, 20, 0.9, 0), (60, 55, 20, 20, 0.7, 1), (52, 50, 20, 20, 0.6, 0)]
        )
        assert apply_ensemble_nms(raw) == apply_ensemble_nms(torch.from_numpy(raw))

    def test_half_precision_input_is_accepted(self) -> None:
        raw = _make_raw([(50, 50, 20, 20, 0.9, 0)])
        [[det]] = apply_ensemble_nms(torch.from_numpy(raw).half())
        assert det['box'] == pytest.approx([40.0, 40.0, 60.0, 60.0])


# --- parity against an independent, deliberately simple reference ---------


def _ref_iou(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ref_nms(
    image: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    agnostic: bool,
    classes: list[int] | None,
) -> list[tuple[list[float], float, int]]:
    """Pure-python greedy NMS, one class at a time, spec-for-spec."""
    cands: list[tuple[list[float], float, int]] = []
    for row in image.astype(np.float64).tolist():
        cx, cy, w, h, obj = row[:5]
        if not obj > conf_thres:
            continue
        scores = [s * obj for s in row[5:]]
        best = max(range(len(scores)), key=lambda k: (scores[k], -k))
        if not scores[best] > conf_thres:
            continue
        if classes is not None and best not in classes:
            continue
        cands.append(([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], scores[best], best))

    cands.sort(key=lambda c: -c[1])
    kept: list[tuple[list[float], float, int]] = []
    for cand in cands:
        if all(
            (not agnostic and k[2] != cand[2]) or _ref_iou(k[0], cand[0]) <= iou_thres for k in kept
        ):
            kept.append(cand)
    return kept[:max_det]


def _random_raw(rng: np.random.Generator, batch: int, n: int, nc: int) -> np.ndarray:
    raw = np.empty((batch, n, 5 + nc), dtype=np.float32)
    # Clustered centers so plenty of boxes overlap and suppression matters.
    centers = rng.uniform(50, 590, size=(batch, 6, 2))
    pick = rng.integers(0, 6, size=(batch, n))
    for b in range(batch):
        raw[b, :, 0:2] = centers[b, pick[b]] + rng.normal(0, 12, size=(n, 2))
    raw[..., 2:4] = rng.uniform(10, 120, size=(batch, n, 2))
    # Distinct, well-separated scores so ordering has no float ties.
    raw[..., 4] = rng.uniform(0, 1, size=(batch, n))
    raw[..., 5:] = rng.uniform(0, 1, size=(batch, n, nc))
    return raw


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('agnostic', [False, True])
def test_parity_with_reference_nms(seed: int, agnostic: bool) -> None:
    rng = np.random.default_rng(seed)
    raw = _random_raw(rng, batch=3, n=120, nc=4)
    conf, iou = 0.2, 0.45
    classes = [0, 2] if seed % 3 == 0 else None
    max_det = 25 if seed % 2 else 300

    got = apply_ensemble_nms(
        raw, conf_thres=conf, iou_thres=iou, max_det=max_det, classes=classes, agnostic=agnostic
    )
    assert len(got) == raw.shape[0]
    for b in range(raw.shape[0]):
        ref = _ref_nms(raw[b], conf, iou, max_det, agnostic, classes)
        assert len(got[b]) == len(ref), f'image {b}'
        got_sorted = sorted(got[b], key=lambda d: -d['score'])
        for det, (rbox, rscore, rcls) in zip(got_sorted, ref, strict=True):
            assert det['class_id'] == rcls
            assert det['score'] == pytest.approx(rscore, rel=1e-5)
            assert det['box'] == pytest.approx(rbox, abs=1e-3)

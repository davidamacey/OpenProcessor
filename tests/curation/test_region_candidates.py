"""W8.4/W8.15: :func:`select_region_candidates` -- floor, NMS, cap, order."""

from __future__ import annotations

from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.region_candidates import select_region_candidates


def _c(x1: float, y1: float, x2: float, y2: float, score: float) -> RegionCandidate:
    return RegionCandidate(bbox_norm=(x1, y1, x2, y2), score=score, source='test')


def test_floor_drops_low_score_candidates() -> None:
    low = _c(0.0, 0.0, 0.1, 0.1, 0.1)
    high = _c(0.5, 0.5, 0.6, 0.6, 0.9)
    result = select_region_candidates([low, high], min_score=0.3, iou=0.5, max_n=4)
    assert result.selected == [high]
    assert (low, 'below_min_score') in result.dropped


def test_nms_drops_high_overlap_keeps_low_overlap() -> None:
    a = _c(0.0, 0.0, 0.5, 0.5, 0.9)
    # b overlaps a at IoU ~0.8 (nearly identical box, slightly shifted)
    b = _c(0.02, 0.02, 0.52, 0.52, 0.8)
    # c overlaps a at IoU ~0.3 (partial overlap)
    c = _c(0.35, 0.0, 0.85, 0.5, 0.7)
    result = select_region_candidates([a, b, c], min_score=0.0, iou=0.5, max_n=4)
    selected_boxes = {s.bbox_norm for s in result.selected}
    assert a.bbox_norm in selected_boxes
    assert b.bbox_norm not in selected_boxes
    assert c.bbox_norm in selected_boxes
    assert (b, 'nms') in result.dropped


def test_max_n_cuts_the_lowest_scoring_survivors() -> None:
    cands = [_c(i * 0.1, 0.0, i * 0.1 + 0.05, 0.05, 1.0 - i * 0.1) for i in range(5)]
    result = select_region_candidates(cands, min_score=0.0, iou=0.1, max_n=2)
    assert len(result.selected) == 2
    assert result.selected[0].score >= result.selected[1].score
    assert sum(1 for _c2, reason in result.dropped if reason == 'over_max') == 3


def test_deterministic_tie_order() -> None:
    # Equal score, no overlap: ties break by (x1, y1).
    first = _c(0.0, 0.0, 0.1, 0.1, 0.5)
    second = _c(0.5, 0.5, 0.6, 0.6, 0.5)
    result = select_region_candidates([second, first], min_score=0.0, iou=0.1, max_n=4)
    assert result.selected == [first, second]


def test_n1_returns_top_scoring_box_pre_w8_behaviour() -> None:
    weak = _c(0.0, 0.0, 0.2, 0.2, 0.4)
    strong = _c(0.6, 0.6, 0.8, 0.8, 0.95)
    result = select_region_candidates([weak, strong], min_score=0.0, iou=0.5, max_n=1)
    assert result.selected == [strong]

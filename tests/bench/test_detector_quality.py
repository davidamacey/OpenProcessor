"""Detector quality against ground truth on synthetic boxes (pure logic, no engine, no images)."""

from __future__ import annotations

import pytest

from scripts.bench import detector_quality as dq


CATS = [1, 2]


def _gt(boxes: dict[int, list[tuple[float, float, float, float, int]]]) -> dq.GroundTruth:
    return dq.GroundTruth(
        boxes={
            img: [dq.GtBox(b[:4], b[4], crowd=False) for b in items] for img, items in boxes.items()
        },
        sizes=dict.fromkeys(boxes, (100, 100)),
    )


def test_iou_known_values() -> None:
    assert dq.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert dq.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert dq.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_operating_point_counts_tp_fp_fn() -> None:
    gt = _gt({1: [(0, 0, 10, 10, 1), (50, 50, 60, 60, 2)]})
    dets: dq.Dets = {
        1: [
            (0, 0, 10, 10, 0.9, 1),  # TP
            (70, 70, 80, 80, 0.8, 1),  # FP, no GT there
            (50, 50, 60, 60, 0.3, 2),  # below the confidence floor: not counted
        ]
    }
    op = dq.operating_point(gt, dets, conf=0.4, iou_thr=0.5, cat_ids=CATS)
    assert (op['tp'], op['fp'], op['fn']) == (1, 1, 1)
    assert op['precision'] == pytest.approx(0.5)
    assert op['recall'] == pytest.approx(0.5)


def test_operating_point_does_not_match_across_classes() -> None:
    gt = _gt({1: [(0, 0, 10, 10, 1)]})
    dets: dq.Dets = {1: [(0, 0, 10, 10, 0.9, 2)]}
    op = dq.operating_point(gt, dets, conf=0.4, iou_thr=0.5, cat_ids=CATS)
    assert (op['tp'], op['fp'], op['fn']) == (0, 1, 1)


def test_operating_point_ignores_detections_inside_crowd_regions() -> None:
    gt = dq.GroundTruth(
        boxes={1: [dq.GtBox((0, 0, 100, 100), 1, crowd=True)]}, sizes={1: (100, 100)}
    )
    op = dq.operating_point(
        gt, {1: [(10, 10, 30, 30, 0.9, 1)]}, conf=0.4, iou_thr=0.5, cat_ids=CATS
    )
    assert (op['tp'], op['fp'], op['fn']) == (0, 0, 0)


def test_coco_ap_perfect_and_empty() -> None:
    gt = _gt({1: [(0, 0, 10, 10, 1)], 2: [(20, 20, 40, 40, 2)]})
    perfect: dq.Dets = {1: [(0, 0, 10, 10, 0.9, 1)], 2: [(20, 20, 40, 40, 0.8, 2)]}
    res = dq.coco_ap(gt, perfect, CATS)
    assert res['ap50'] == pytest.approx(1.0)
    assert res['ap'] == pytest.approx(1.0)
    assert res['per_class'][1]['ap50'] == pytest.approx(1.0)
    empty = dq.coco_ap(gt, {}, CATS)
    assert empty['ap'] == 0.0
    assert empty['ap50'] == 0.0


def test_coco_ap_drops_with_a_loose_box() -> None:
    gt = _gt({1: [(0, 0, 10, 10, 1)]})
    exact = dq.coco_ap(gt, {1: [(0, 0, 10, 10, 0.9, 1)]}, [1])
    loose = dq.coco_ap(gt, {1: [(0, 0, 10, 14, 0.9, 1)]}, [1])
    assert loose['ap50'] == pytest.approx(1.0)
    assert loose['ap'] < exact['ap']


def test_compare_moved_box_is_scored_against_ground_truth() -> None:
    gt = _gt({1: [(0, 0, 100, 100, 1), (200, 200, 300, 300, 2)]})
    a: dq.Dets = {
        1: [
            (0, 0, 100, 100, 0.9, 1),  # B moves it away from GT: worse
            (200, 200, 290, 300, 0.9, 2),  # B moves it onto GT: better
        ]
    }
    b: dq.Dets = {1: [(0, 0, 90, 100, 0.9, 1), (200, 200, 300, 300, 0.9, 2)]}
    res = dq.compare(gt, a, b, conf=0.4)
    assert res['matched'] == 2
    assert res['moved'] == 2
    assert res['moved_better'] == 1
    assert res['moved_worse'] == 1
    assert res['moved_equal'] == 0
    assert res['mean_gt_iou_delta_moved'] == pytest.approx(
        ((0.9 - 1.0) + (1.0 - 0.9)) / 2, abs=1e-9
    )


def test_compare_identical_boxes_are_not_moved_and_unmatched_are_split_by_truth() -> None:
    gt = _gt({1: [(0, 0, 100, 100, 1), (200, 200, 300, 300, 2)]})
    a: dq.Dets = {1: [(0, 0, 100, 100, 0.9, 1), (200, 200, 300, 300, 0.9, 2)]}
    b: dq.Dets = {1: [(0, 0, 100, 100, 0.9, 1), (500, 500, 600, 600, 0.9, 1)]}
    res = dq.compare(gt, a, b, conf=0.4)
    assert res['matched'] == 1
    assert res['moved'] == 0
    assert res['only_a'] == {'total': 1, 'on_truth': 1, 'off_truth': 0}
    assert res['only_b'] == {'total': 1, 'on_truth': 0, 'off_truth': 1}


def test_category_ids_follow_sorted_coco_ids() -> None:
    cats = [{'id': 3}, {'id': 1}, {'id': 90}]
    assert dq.class_to_category(cats) == [1, 3, 90]


def test_build_ground_truth_selects_only_requested_images() -> None:
    ann = {
        'images': [
            {'id': 1, 'width': 10, 'height': 10},
            {'id': 2, 'width': 20, 'height': 20},
        ],
        'annotations': [
            {'image_id': 1, 'category_id': 5, 'bbox': [1, 2, 3, 4], 'iscrowd': 0},
            {'image_id': 2, 'category_id': 5, 'bbox': [0, 0, 1, 1], 'iscrowd': 1},
        ],
        'categories': [{'id': 5, 'name': 'x'}],
    }
    gt = dq.build_ground_truth(ann, [1])
    assert set(gt.boxes) == {1}
    assert gt.boxes[1][0].xyxy == (1, 2, 4, 6)
    assert gt.sizes == {1: (10, 10)}


def test_merge_annotations_joins_splits_and_keeps_categories() -> None:
    a = {'images': [{'id': 1}], 'annotations': [{'image_id': 1}], 'categories': [{'id': 5}]}
    b = {'images': [{'id': 2}], 'annotations': [{'image_id': 2}], 'categories': [{'id': 5}]}
    merged = dq.merge_annotations([a, b])
    assert [i['id'] for i in merged['images']] == [1, 2]
    assert len(merged['annotations']) == 2
    assert merged['categories'] == [{'id': 5}]

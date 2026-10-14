"""Multi-class decode and NMS, multi-class dataset loading and per-class metrics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from curation.bakeoff_harness._helpers import _BOX, _det, _mc_dataset
from scripts.curation.bakeoff import class_map, metrics, scoring
from scripts.curation.bakeoff.backends.yolo_post import decode_yolo26_e2e, decode_yolo_v11, nms
from scripts.curation.bakeoff.dataset import YoloTestSet


if TYPE_CHECKING:
    from pathlib import Path


# --- multi-class decode / NMS ------------------------------------------------------

# A [1, 5, 6] single-class head and the exact (boxes, scores) the pre-multi-class
# decoder produced for it (captured before the change; scale 0.5, pad (10, 20)).
_NC1_HEAD = np.array(
    [
        [
            [0.5, 0.25, 0.75, 0.1, 0.9, 0.3],
            [0.5, 0.5, 0.25, 0.2, 0.1, 0.8],
            [0.2, 0.1, 0.3, 0.05, 0.4, 0.2],
            [0.2, 0.3, 0.1, 0.05, 0.2, 0.4],
            [0.9, 0.0005, 0.6, 0.3, 0.002, 0.7],
        ]
    ],
    dtype=np.float32,
)
_NC1_GOLDEN_BOXES = [
    [60.0, 40.0, 100.0, 80.0],
    [100.0, 0.0, 160.0, 20.0],
    [-5.0, -5.0, 5.0, 5.0],
    [120.0, -40.0, 200.0, 0.0],
    [20.0, 80.0, 60.0, 160.0],
]
_NC1_GOLDEN_SCORES = [0.9, 0.6, 0.3, 0.002, 0.7]


def _decode(head: np.ndarray):
    return decode_yolo_v11(
        head, scale=0.5, pad=(10, 20), input_size=100, conf_thresh=0.001, coords_normalized=True
    )


def test_yolo_decode_multiclass_head() -> None:
    boxes, scores, class_ids = _decode(_NC1_HEAD)
    np.testing.assert_allclose(boxes, _NC1_GOLDEN_BOXES, rtol=1e-6)
    np.testing.assert_allclose(scores, _NC1_GOLDEN_SCORES, rtol=1e-6)
    assert class_ids.tolist() == [0, 0, 0, 0, 0]
    # Same head transposed ([1, N, 5]) decodes identically.
    boxes_t, scores_t, _ = _decode(_NC1_HEAD.transpose(0, 2, 1))
    np.testing.assert_allclose(boxes_t, _NC1_GOLDEN_BOXES, rtol=1e-6)
    np.testing.assert_allclose(scores_t, _NC1_GOLDEN_SCORES, rtol=1e-6)

    # nc = 3: score = max over the class rows, class = argmax.
    cls_rows = np.array(
        [
            [0.1, 0.7, 0.0005, 0.2, 0.3, 0.1],
            [0.9, 0.2, 0.0001, 0.1, 0.6, 0.2],
            [0.3, 0.1, 0.0002, 0.8, 0.1, 0.95],
        ],
        dtype=np.float32,
    )
    head3 = np.concatenate([_NC1_HEAD[:, :4], cls_rows[None]], axis=1)  # [1, 7, 6]
    # Real heads have thousands of anchors (N >> 4 + nc); pad to 8 anchors with
    # sub-threshold ones so the channel axis is the shorter one, as in practice.
    head3 = np.concatenate([head3, np.zeros((1, 7, 2), dtype=np.float32)], axis=2)
    boxes, scores, class_ids = _decode(head3)
    assert class_ids.tolist() == [1, 0, 2, 1, 2]
    assert scores.tolist() == pytest.approx([0.9, 0.7, 0.8, 0.6, 0.95])
    assert boxes[0].tolist() == pytest.approx(_NC1_GOLDEN_BOXES[0])
    _, _, cls_t = _decode(head3.transpose(0, 2, 1))
    assert cls_t.tolist() == [1, 0, 2, 1, 2]
    # A known class count disambiguates a head with fewer anchors than channels.
    _, _, cls_n = decode_yolo_v11(
        head3[:, :, :6],
        scale=0.5,
        pad=(10, 20),
        input_size=100,
        conf_thresh=0.001,
        coords_normalized=True,
        num_classes=3,
    )
    assert cls_n.tolist() == [1, 0, 2, 1, 2]


def test_decode_yolo26_e2e_keeps_class_column() -> None:
    out = np.array(
        [[[10, 20, 30, 40, 0.9, 3], [0, 0, 5, 5, 0.0001, 1], [50, 60, 70, 80, 0.5, 7]]],
        dtype=np.float32,
    )
    boxes, scores, class_ids = decode_yolo26_e2e(out, scale=0.5, pad=(10, 0), conf_thresh=0.001)
    assert class_ids.tolist() == [3, 7]
    assert class_ids.dtype.kind == 'i'
    assert scores.tolist() == pytest.approx([0.9, 0.5])
    assert boxes[0].tolist() == pytest.approx([0.0, 40.0, 40.0, 80.0])


def test_nms_is_per_class() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    # Class-agnostic: the two lower-scored overlapping boxes are suppressed.
    assert nms(boxes, scores, 0.5) == [0]
    # Per class: box 1 is another class, so it survives; box 2 is suppressed by box 0.
    assert nms(boxes, scores, 0.5, class_ids=np.array([0, 1, 0])) == [0, 1]


# --- multi-class dataset, class mapping, metrics -----------------------------------------


def test_dataset_loads_all_classes_and_present_set(tmp_path: Path) -> None:
    names = [f'c{i}' for i in range(84)]
    root = _mc_dataset(
        tmp_path / 'ds',
        {'a': [(37, *_BOX), (43, 0.8, 0.8, 0.1, 0.1)], 'b': [(37, *_BOX)], 'bg': []},
        names,
    )
    ds = YoloTestSet.from_dataset_root(root)
    assert ds.present_class_ids == {37, 43}
    assert ds.n_gt_by_class == {37: 2, 43: 1}
    assert ds.class_names[37] == 'c37'
    assert (ds.n_positive_frames, ds.n_background_frames) == (2, 1)
    gt = ds.coco_gt([37, 43])
    assert gt['categories'] == [{'id': 38, 'name': 'c37'}, {'id': 44, 'name': 'c43'}]
    assert sorted(a['category_id'] for a in gt['annotations']) == [38, 38, 44]
    only37 = ds.coco_gt([37])
    assert [a['category_id'] for a in only37['annotations']] == [38, 38]

    # data.yaml names: inline JSON list (multi-class export) and block dict
    # (single-class export) both parse.
    assert class_map.read_names(root / 'data.yaml')[43] == 'c43'
    block = tmp_path / 'block.yaml'
    block.write_text("path: /x\nnc: 2\nnames:\n  0: Mini Cooper\n  1: 'vw'\n")
    assert class_map.read_names(block) == {0: 'Mini Cooper', 1: 'vw'}
    (root / 'class_registry.json').write_text(json.dumps({'export_id_map': {'38': 37}}))
    assert class_map.read_export_id_map(root) == {38: 37}
    assert class_map.read_export_id_map(tmp_path) == {}


def test_metrics_per_class_perfect_and_miss(tmp_path: Path) -> None:
    ds = YoloTestSet.from_dataset_root(
        _mc_dataset(tmp_path / 'ds', {'a': [(0, *_BOX)], 'b': [(1, *_BOX)]}, ['A', 'B'])
    )
    a, b = ds.images
    dets = {a.image_id: [_det(0)], b.image_id: []}
    res = metrics.coco_eval(ds.coco_gt([0, 1]), metrics.detections_to_coco(dets, None), [0, 1])
    assert res.per_class[0].ap50 == pytest.approx(1.0)
    assert res.per_class[1].ap50 == pytest.approx(0.0)
    assert res.overall.map_50 == pytest.approx(0.5)
    op = metrics.operating_point(ds.images, dets, conf=0.25, iou=0.45, class_ids=[0, 1])
    assert (op.per_class[0].tp, op.per_class[0].fp, op.per_class[0].fn) == (1, 0, 0)
    assert (op.per_class[1].tp, op.per_class[1].fp, op.per_class[1].fn) == (0, 0, 1)
    assert (op.micro.tp, op.micro.fp, op.micro.fn) == (1, 0, 1)
    assert op.micro.precision == pytest.approx(1.0)
    assert op.micro.recall == pytest.approx(0.5)
    # A class-0 box on the class-1 frame is an FP for class 0, not a TP for class 1.
    op = metrics.operating_point(
        ds.images, {b.image_id: [_det(0)]}, conf=0.25, iou=0.45, class_ids=[0, 1]
    )
    assert (op.per_class[0].fp, op.per_class[1].tp) == (1, 0)


def test_overall_map_equals_cocoeval_stats(tmp_path: Path) -> None:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    labels = {
        'a': [(0, *_BOX), (1, 0.8, 0.8, 0.2, 0.2)],
        'b': [(1, *_BOX), (2, 0.2, 0.8, 0.2, 0.3)],
        'c': [(2, *_BOX), (0, 0.75, 0.25, 0.3, 0.3)],
        'd': [],
    }
    ds = YoloTestSet.from_dataset_root(_mc_dataset(tmp_path / 'ds', labels, ['x', 'y', 'z']))
    ids = {img.path.stem: img.image_id for img in ds.images}
    dets = {
        ids['a']: [_det(0, 0.9), _det(1, 0.4, (68, 68, 90, 92)), _det(2, 0.3)],
        ids['b']: [_det(1, 0.8, (22, 18, 61, 63)), _det(0, 0.6)],
        ids['c']: [_det(2, 0.7), _det(0, 0.5, (60, 10, 88, 40)), _det(1, 0.2, (5, 5, 30, 30))],
        ids['d']: [_det(0, 0.35)],
    }
    gt = ds.coco_gt([0, 1, 2])
    results = metrics.detections_to_coco(dets, None)
    res = metrics.coco_eval(gt, results, [0, 1, 2])

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(results), iouType='bbox')
        ev.params.catIds = [1, 2, 3]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    assert res.overall.map_50_95 == pytest.approx(ev.stats[0], abs=1e-9)
    assert res.overall.map_50 == pytest.approx(ev.stats[1], abs=1e-9)
    assert res.overall.map_75 == pytest.approx(ev.stats[2], abs=1e-9)
    per_class_mean = sum(res.per_class[k].ap50_95 for k in (0, 1, 2)) / 3
    assert per_class_mean == pytest.approx(ev.stats[0], abs=1e-9)
    assert 0.0 < res.overall.map_50 < 1.0  # a non-trivial scenario


def _score(ds: YoloTestSet, dets, mapping: class_map.ClassMapping) -> dict:
    return scoring.score(ds, dets, mapping, scored_class_ids=sorted(ds.present_class_ids))


def test_uncovered_class_reported_not_dropped(tmp_path: Path) -> None:
    ds = YoloTestSet.from_dataset_root(
        _mc_dataset(tmp_path / 'ds', {'a': [(0, *_BOX)], 'b': [(1, *_BOX)]}, ['A', 'B'])
    )
    model_names = {0: 'a'}
    mapping = class_map.finalize(
        'names',
        class_map.resolve_by_names(model_names, ds.class_names),
        model_names=model_names,
        eval_names=ds.class_names,
        scored_class_ids=[0, 1],
    )
    blocks = _score(ds, {ds.images[0].image_id: [_det(0)]}, mapping)
    rows = {r['eval_class_id']: r for r in blocks['per_class']}
    assert rows[0]['covered'] is True
    assert rows[0]['model_class_ids'] == [0]
    assert rows[0]['ap50'] == pytest.approx(1.0)
    assert rows[1]['covered'] is False
    assert rows[1]['model_class_ids'] == []
    assert rows[1]['n_gt'] == 1
    for key in ('ap50', 'ap50_95', 'ap75', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'):
        assert rows[1][key] is None, key
    assert blocks['coverage']['not_covered'] == [{'eval_class_id': 1, 'name': 'B'}]
    assert blocks['coverage']['n_covered'] == 1
    assert blocks['coverage']['n_eval_classes'] == 2
    assert blocks['overall']['n_classes'] == 1
    assert blocks['overall']['map_50'] == pytest.approx(1.0)  # B's miss is not charged
    assert blocks['overall']['fn'] == 0


def test_unmapped_model_class_counted(tmp_path: Path) -> None:
    ds = YoloTestSet.from_dataset_root(
        _mc_dataset(tmp_path / 'ds', {'a': [(0, *_BOX)], 'b': []}, ['A', 'B'])
    )
    model_names = {0: 'A', 1: 'zebra', 2: 'b'}
    mapping = class_map.finalize(
        'names',
        class_map.resolve_by_names(model_names, ds.class_names),
        model_names=model_names,
        eval_names=ds.class_names,
        scored_class_ids=[0],
    )
    a, bg = ds.images
    dets = {
        a.image_id: [_det(0), _det(1, 0.8), _det(1, 0.7)],
        bg.image_id: [_det(1, 0.6), _det(2, 0.9)],  # class 2 -> 'B', not scored
    }
    blocks = scoring.score(ds, dets, mapping, scored_class_ids=[0])
    cov = blocks['coverage']
    assert cov['unmapped_model_classes'] == [
        {'model_class_id': 1, 'name': 'zebra', 'n_predictions': 3}
    ]
    assert cov['predictions_outside_scored_classes'] == 1
    assert blocks['overall']['fp'] == 0  # unmapped/outside predictions are not scored
    assert blocks['overall']['map_50'] == pytest.approx(1.0)

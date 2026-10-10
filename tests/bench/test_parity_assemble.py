"""Ground truth assignment, crop rectangles and the array assembly."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts.bench import parity_assemble as pa, parity_store as ps


if TYPE_CHECKING:
    from pathlib import Path


def test_crop_rect_rounds_and_keeps_at_least_one_pixel() -> None:
    assert pa.crop_rect([0.1, 0.2, 0.5, 0.9], 100, 50) == [10, 10, 50, 45]
    assert pa.crop_rect([0.5, 0.5, 0.5, 0.5], 10, 10) == [5, 5, 6, 6]
    assert pa.crop_rect([-0.2, 0.0, 0.3, 1.0], 100, 100)[0] == 0


def test_assign_gt_takes_best_overlap_and_none_below_half() -> None:
    gt = (
        np.array([[0.0, 0.0, 0.5, 0.5], [0.45, 0.0, 1.0, 0.5]], np.float32),
        ['person', 'dog'],
    )
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5], [0.0, 0.6, 0.2, 0.9]], np.float32)
    assert pa.assign_gt(boxes, gt) == ['person', 'dog', 'none']
    assert pa.assign_gt(boxes, (np.zeros((0, 4), np.float32), [])) == ['none'] * 3


def _write_coco(root: Path) -> dict:
    ann_dir = root / 'annotations'
    ann_dir.mkdir()
    cats = [{'id': 1, 'name': 'person'}, {'id': 3, 'name': 'car'}]
    for split, image_id, cat in (('val2017', 7, 1), ('train2017', 9, 3)):
        doc = {
            'categories': cats,
            'annotations': [
                {'image_id': image_id, 'category_id': cat, 'bbox': [10, 20, 30, 40], 'iscrowd': 0},
                {'image_id': image_id, 'category_id': cat, 'bbox': [0, 0, 5, 5], 'iscrowd': 1},
                {'image_id': 999, 'category_id': cat, 'bbox': [0, 0, 5, 5], 'iscrowd': 0},
            ],
        }
        (ann_dir / f'instances_{split}.json').write_text(json.dumps(doc))
    return {
        'images': [
            {'image_id': 7, 'file_name': 'a.jpg', 'split': 'val2017', 'width': 100, 'height': 200},
            {'image_id': 9, 'file_name': 'b.jpg', 'split': 'train2017', 'width': 50, 'height': 100},
            {'image_id': 11, 'file_name': 'c.jpg', 'split': 'val2017', 'width': 10, 'height': 10},
        ]
    }


def test_load_gt_normalises_boxes_and_drops_crowd(tmp_path: Path) -> None:
    pin = _write_coco(tmp_path)
    gt = pa.load_gt(tmp_path, pin)
    boxes, names = gt['a.jpg']
    assert names == ['person']
    assert boxes[0].tolist() == pytest.approx([0.1, 0.1, 0.4, 0.3])
    assert gt['b.jpg'][1] == ['car']
    assert gt['c.jpg'][1] == []


def test_load_gt_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='instances_val2017'):
        pa.load_gt(tmp_path, {'images': [{'image_id': 1, 'file_name': 'a', 'split': 'val2017'}]})


def _item(item_id: str, image_id: str, box: list[float], **extra) -> dict:
    return {
        'id': item_id,
        'image_id': image_id,
        'bbox_norm': box,
        'confidence': 0.5,
        'detector_confidence': 0.9,
        'detector_class_name': 'cat',
        'class_name': 'cat',
        'cluster_id': None,
        **extra,
    }


def test_build_arrays_sorts_by_file_then_box_and_marks_missing_vectors() -> None:
    items = [
        _item('b1', 'ib', [0.5, 0.1, 0.9, 0.5]),
        _item('a2', 'ia', [0.5, 0.1, 0.9, 0.5], cluster_id=3, cluster_is_core=True),
        _item('a1', 'ia', [0.1, 0.1, 0.4, 0.5], cluster_distance=0.2),
    ]
    arrays = pa.build_arrays(
        items,
        image_names={'ia': 'a.jpg', 'ib': 'b.jpg'},
        image_dims={'a.jpg': (100, 100), 'b.jpg': (200, 100)},
        vectors={'a1': [1.0, 0.0], 'a2': [0.0, 1.0]},
        frame_vectors={'ia': [1.0, 1.0]},
        gt=None,
        vlm={'a2': 'dog'},
        probes={'ocr': [('b.jpg', [0, 0, 1, 1], 'HI')]},
    )
    ps.validate(arrays)
    assert arrays['item_id'].tolist() == ['a1', 'a2', 'b1']
    assert arrays['item_has_vec'].tolist() == [True, True, False]
    assert arrays['item_cluster'].tolist() == [-1, 3, -1]
    assert arrays['item_cluster_core'].tolist() == [-1, 1, -1]
    assert np.isnan(arrays['item_cluster_distance'][1])
    assert arrays['item_vlm'].tolist() == ['', 'dog', '']
    assert arrays['item_gt'].tolist() == ['', '', '']
    assert arrays['frame_image'].tolist() == ['a.jpg']
    assert arrays['ocr_text'].tolist() == ['HI']
    assert arrays['item_vec'].dtype == np.float16


def test_build_arrays_records_ground_truth_per_item() -> None:
    gt = {'a.jpg': (np.array([[0.1, 0.1, 0.4, 0.5]], np.float32), ['cat'])}
    arrays = pa.build_arrays(
        [_item('a1', 'ia', [0.1, 0.1, 0.4, 0.5]), _item('a2', 'ia', [0.6, 0.6, 0.9, 0.9])],
        image_names={'ia': 'a.jpg'},
        image_dims={'a.jpg': (10, 10)},
        vectors={},
        frame_vectors={},
        gt=gt,
        vlm={},
        probes={},
    )
    assert arrays['item_gt'].tolist() == ['cat', 'none']


def test_build_arrays_empty_capture_is_valid() -> None:
    arrays = pa.build_arrays(
        [], image_names={}, image_dims={}, vectors={}, frame_vectors={}, gt=None, vlm={}, probes={}
    )
    ps.validate(arrays)
    assert len(arrays['item_id']) == 0

"""W10.2.1/W10.17: COCO dataset scanning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.services.curation.dataset_import.coco import CocoAnnotationFile, scan_coco


if TYPE_CHECKING:
    from pathlib import Path


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def _base_data(**overrides) -> dict:
    data = {
        'images': [{'id': 1, 'file_name': 'a.jpg', 'width': 100, 'height': 100}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 10, 'bbox': [10, 10, 20, 20], 'iscrowd': 0}
        ],
        'categories': [{'id': 10, 'name': 'car'}],
    }
    data.update(overrides)
    return data


def test_pixel_to_normalized(tmp_path: Path) -> None:
    ann_path = _write(tmp_path, 'instances_train.json', _base_data())
    scan = scan_coco([CocoAnnotationFile(path=ann_path, images_dir=tmp_path)])
    assert len(scan.entries) == 1
    entry = scan.entries[0]
    assert entry.label_state == 'labeled'
    box = entry.boxes[0]
    assert box.dataset_class == 'car'
    assert box.bbox_norm == (0.1, 0.1, 0.3, 0.3)


def test_crowd_skipped(tmp_path: Path) -> None:
    data = _base_data(
        annotations=[
            {'id': 1, 'image_id': 1, 'category_id': 10, 'bbox': [1, 1, 2, 2], 'iscrowd': 1}
        ]
    )
    ann_path = _write(tmp_path, 'instances_train.json', data)
    scan = scan_coco([CocoAnnotationFile(path=ann_path, images_dir=tmp_path)])
    codes = {i.code for i in scan.issues.issues()}
    assert 'coco_crowd_skipped' in codes
    assert scan.entries[0].label_state == 'negative'


def test_duplicate_category_names_blocking(tmp_path: Path) -> None:
    data = _base_data(categories=[{'id': 10, 'name': 'car'}, {'id': 11, 'name': 'car'}])
    ann_path = _write(tmp_path, 'instances_train.json', data)
    scan = scan_coco([CocoAnnotationFile(path=ann_path, images_dir=tmp_path)])
    codes = [i.code for i in scan.issues.issues()]
    assert 'coco_category_duplicate_name' in codes
    from src.services.curation.dataset_import.issues import ISSUE_CATALOG

    assert ISSUE_CATALOG['coco_category_duplicate_name'].blocking


def test_no_annotations_is_negative(tmp_path: Path) -> None:
    data = _base_data(annotations=[])
    ann_path = _write(tmp_path, 'instances_val.json', data)
    scan = scan_coco([CocoAnnotationFile(path=ann_path, images_dir=tmp_path)])
    assert scan.entries[0].label_state == 'negative'


def test_split_from_file_name(tmp_path: Path) -> None:
    ann_path = _write(tmp_path, 'instances_val.json', _base_data())
    scan = scan_coco([CocoAnnotationFile(path=ann_path, images_dir=tmp_path)])
    assert scan.entries[0].split == 'val'


def test_never_by_index_category_id_ignored(tmp_path: Path) -> None:
    """category_id 10 maps to class name 'car' regardless of its numeric
    value — never compared with a registry id."""
    data = _base_data(
        categories=[{'id': 999, 'name': 'car'}],
        annotations=[
            {'id': 1, 'image_id': 1, 'category_id': 999, 'bbox': [10, 10, 20, 20], 'iscrowd': 0}
        ],
    )
    ann_path = _write(tmp_path, 'instances_train.json', data)
    scan = scan_coco([CocoAnnotationFile(path=ann_path, images_dir=tmp_path)])
    assert scan.entries[0].boxes[0].dataset_class == 'car'

"""The ``fetch_coco_subset.py`` output imports as written: ``images/`` plus one
``annotations/instances_<split>.json`` per split, so the importer's file-name
split inference gives val frames the ``val`` split instead of pinning all to train."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from PIL import Image

from scripts.datasets import bench_set, coco_fetch
from src.services.curation.dataset_import.options import DatasetSource
from src.services.curation.dataset_import.scan import scan_dataset


if TYPE_CHECKING:
    from pathlib import Path


def _row(image_id: int, split: str) -> dict[str, Any]:
    return {
        'image_id': image_id,
        'file_name': f'{image_id:012d}.jpg',
        'width': 32,
        'height': 32,
        'split': split,
    }


def test_the_fetch_output_scans_with_val_and_train_splits(tmp_path: Path) -> None:
    rows = [_row(1, 'val2017'), _row(2, 'val2017'), _row(3, 'train2017')]
    (tmp_path / 'images').mkdir()
    for r in rows:
        Image.new('RGB', (32, 32)).save(tmp_path / 'images' / r['file_name'])
    anns = {
        r['image_id']: [
            {'id': r['image_id'], 'image_id': r['image_id'], 'category_id': 3,
             'bbox': [2, 2, 10, 10], 'iscrowd': 0}
        ]
        for r in rows
    }  # fmt: skip
    coco_fetch.write_import_annotations(tmp_path, rows, anns, [{'id': 3, 'name': 'car'}])
    scan = scan_dataset(DatasetSource(path=str(tmp_path)), path_guard=lambda _p: True)
    assert scan.format == 'coco'
    by_split = {e.rel_path: e.split for e in scan.entries}
    assert sorted(str(v) for v in by_split.values()) == ['train', 'val', 'val']
    assert by_split['000000000003.jpg'] == 'train'


def test_manifest_rows_reads_both_pin_shapes(tmp_path: Path) -> None:
    rows = [{'image_id': 1, 'file_name': 'a.jpg', 'split': 'val2017'}]
    plain, pin = tmp_path / 'plain.json', tmp_path / 'pin.json'
    plain.write_text(json.dumps(rows))
    pin.write_text(json.dumps({'schema': 1, 'images': rows}))
    assert bench_set.manifest_rows(plain) == rows
    assert bench_set.manifest_rows(pin) == rows

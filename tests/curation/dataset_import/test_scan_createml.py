"""CreateML JSON dataset scanning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.annotation_fixtures import cml_ann, write_createml
from curation.dataset_import.harness import write_image
from src.services.curation.dataset_import import createml
from src.services.curation.dataset_import.createml import scan_createml
from src.services.curation.dataset_import.issues import ISSUE_CATALOG
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError
from src.services.curation.dataset_import.scan import detect_format


if TYPE_CHECKING:
    from pathlib import Path


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def test_detected_and_center_pixels_become_normalized_corners(tmp_path: Path) -> None:
    write_createml(
        tmp_path,
        {'a.jpg': [cml_ann('dog', 30, 50, 40, 60), cml_ann('person', 75, 25, 30, 30)]},
    )
    assert detect_format(tmp_path) == 'createml'
    scan = scan_createml(tmp_path)
    assert scan.format == 'createml'
    boxes = scan.entries[0].boxes
    assert [b.dataset_class for b in boxes] == ['dog', 'person']
    assert boxes[0].bbox_norm == pytest.approx((0.10, 0.20, 0.50, 0.80))
    assert scan.class_ids == {}
    assert scan.class_box_counts == {'dog': 1, 'person': 1}


def test_no_annotations_is_negative(tmp_path: Path) -> None:
    write_createml(tmp_path, {'a.jpg': []})
    assert detect_format(tmp_path) == 'createml'
    assert scan_createml(tmp_path).entries[0].label_state == 'negative'


def test_split_directories_set_the_split(tmp_path: Path) -> None:
    write_createml(tmp_path / 'train', {'a.jpg': [cml_ann('dog', 30, 50, 40, 60)]})
    write_createml(tmp_path / 'valid', {'b.jpg': [cml_ann('dog', 30, 50, 40, 60)]}, size=(110, 100))
    assert detect_format(tmp_path) == 'createml'
    by = {e.source_stem: e.split for e in scan_createml(tmp_path).entries}
    assert by == {'a': 'train', 'b': 'val'}


@pytest.mark.parametrize(
    ('ann', 'code'),
    [
        (cml_ann('dog', 95, 50, 30, 20), 'label_coords_out_of_range'),
        (cml_ann('dog', 50, 50, 0, 20), 'label_box_degenerate'),
    ],
)
def test_bad_boxes_are_issues(tmp_path: Path, ann, code: str) -> None:
    write_createml(tmp_path, {'a.jpg': [ann]})
    scan = scan_createml(tmp_path)
    assert code in _codes(scan)
    assert scan.entries[0].boxes == []


def test_malformed_annotations_are_skipped(tmp_path: Path) -> None:
    anns = [
        cml_ann('dog', 30, 50, 40, 60),
        {'label': 'x', 'coordinates': {'x': 'a', 'y': 1, 'width': 1, 'height': 1}},
        {'label': 'x'},
        {'label': '', 'coordinates': {'x': 5, 'y': 5, 'width': 4, 'height': 4}},
        'junk',
    ]
    write_createml(tmp_path, {'a.jpg': anns})
    scan = scan_createml(tmp_path)
    assert len(scan.entries[0].boxes) == 1
    assert [i.count for i in scan.issues.issues() if i.code == 'label_row_malformed'] == [4]


@pytest.mark.parametrize('ref', ['../../x.jpg', '/etc/hostname', '..\\x.jpg'])
def test_escaping_image_refs_are_refused(tmp_path: Path, ref: str) -> None:
    write_image(tmp_path.parent / 'x.jpg')
    write_createml(tmp_path / 'ds', {'a.jpg': [cml_ann('dog', 30, 50, 40, 60)]})
    path = tmp_path / 'ds/_annotations.createml.json'
    path.write_text(path.read_text().replace('"a.jpg"', json.dumps(ref)))
    scan = scan_createml(tmp_path / 'ds')
    assert scan.entries == []
    assert 'image_file_missing' in _codes(scan)


@pytest.mark.parametrize('body', ['not json', '{"image": "a.jpg"}', '[1]', '[{"image": 3}]'])
def test_malformed_file_is_a_blocking_issue(tmp_path: Path, body: str) -> None:
    write_createml(tmp_path, {'a.jpg': []})
    (tmp_path / '_annotations.createml.json').write_text(body)
    scan = scan_createml(tmp_path)
    assert 'createml_json_invalid' in _codes(scan)
    assert ISSUE_CATALOG['createml_json_invalid'].blocking
    assert scan.entries == []


def test_oversize_file_is_blocking(tmp_path: Path, monkeypatch) -> None:
    write_createml(tmp_path, {'a.jpg': []})
    monkeypatch.setattr(createml, 'MAX_CREATEML_JSON_BYTES', 8)
    scan = scan_createml(tmp_path)
    assert 'dataset_file_too_large' in _codes(scan)
    assert scan.entries == []


def test_too_many_records_is_refused(tmp_path: Path, monkeypatch) -> None:
    write_createml(tmp_path, {'a.jpg': [], 'b.jpg': []})
    monkeypatch.setenv('OP_DATASET_PREVIEW_MAX_FILES', '1')
    assert 'dataset_too_large' in _codes(scan_createml(tmp_path))


def test_path_guard_refusal_raises(tmp_path: Path) -> None:
    write_createml(tmp_path, {'a.jpg': []})
    with pytest.raises(DatasetPathNotAllowedError):
        scan_createml(tmp_path, path_guard=lambda _p: False)

"""LabelMe JSON dataset scanning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.annotation_fixtures import lm_rect, write_labelme
from curation.dataset_import.harness import write_image
from src.services.curation.dataset_import import labelme
from src.services.curation.dataset_import.issues import ISSUE_CATALOG
from src.services.curation.dataset_import.labelme import scan_labelme
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError
from src.services.curation.dataset_import.scan import detect_format


if TYPE_CHECKING:
    from pathlib import Path


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def test_detected_and_corners_in_any_order_are_normalized(tmp_path: Path) -> None:
    shapes = [lm_rect('dog', (10, 20), (50, 80)), lm_rect('person', (90, 40), (60, 10))]
    write_labelme(tmp_path, {'a': shapes})
    assert detect_format(tmp_path) == 'labelme'
    scan = scan_labelme(tmp_path)
    assert scan.format == 'labelme'
    boxes = scan.entries[0].boxes
    assert [b.dataset_class for b in boxes] == ['dog', 'person']
    assert boxes[0].bbox_norm == pytest.approx((0.10, 0.20, 0.50, 0.80))
    assert boxes[1].bbox_norm == pytest.approx((0.60, 0.10, 0.90, 0.40))
    assert scan.class_ids == {}
    assert scan.entries[0].declared_size == (100, 100)


def test_empty_shapes_is_negative_and_polygon_only_is_unlabeled(tmp_path: Path) -> None:
    poly = {'label': 'dog', 'points': [[1, 1], [9, 1], [5, 9]], 'shape_type': 'polygon'}
    write_labelme(tmp_path, {'neg': [], 'poly': [poly]})
    scan = scan_labelme(tmp_path)
    by = {e.source_stem: e.label_state for e in scan.entries}
    assert by == {'neg': 'negative', 'poly': 'unlabeled'}
    assert 'shape_not_box_skipped' in _codes(scan)


def test_missing_declared_size_is_read_from_the_image(tmp_path: Path) -> None:
    write_labelme(
        tmp_path, {'a': [lm_rect('dog', (20, 0), (100, 50))]}, size=(200, 100), declare_size=False
    )
    scan = scan_labelme(tmp_path)
    assert scan.entries[0].boxes[0].bbox_norm == pytest.approx((0.10, 0.0, 0.50, 0.50))
    assert scan.entries[0].declared_size is None


def test_wrong_declared_size_skips_the_image(tmp_path: Path) -> None:
    write_labelme(tmp_path, {'a': [lm_rect('dog', (1, 1), (20, 20))]})
    path = tmp_path / 'a.json'
    doc = json.loads(path.read_text())
    doc['imageWidth'] = 640
    path.write_text(json.dumps(doc))
    scan = scan_labelme(tmp_path)
    assert scan.entries == []
    assert 'image_size_mismatch' in _codes(scan)


def test_split_directories_set_the_split(tmp_path: Path) -> None:
    write_labelme(tmp_path / 'train', {'a': [lm_rect('dog', (1, 1), (20, 20))]})
    write_labelme(tmp_path / 'val', {'b': [lm_rect('dog', (1, 1), (20, 20))]}, size=(110, 100))
    assert detect_format(tmp_path) == 'labelme'
    by = {e.source_stem: e.split for e in scan_labelme(tmp_path).entries}
    assert by == {'a': 'train', 'b': 'val'}


@pytest.mark.parametrize(
    ('p1', 'p2', 'code'),
    [((1, 1), (150, 50), 'label_coords_out_of_range'), ((10, 5), (10, 50), 'label_box_degenerate')],
)
def test_bad_boxes_are_issues(tmp_path: Path, p1, p2, code: str) -> None:
    write_labelme(tmp_path, {'a': [lm_rect('dog', p1, p2)]})
    scan = scan_labelme(tmp_path)
    assert code in _codes(scan)
    assert scan.entries[0].boxes == []


def test_malformed_rectangles_are_skipped(tmp_path: Path) -> None:
    shapes = [
        lm_rect('dog', (1, 1), (20, 20)),
        {'label': 'x', 'shape_type': 'rectangle', 'points': [[1, 1]]},
        {'label': 'x', 'shape_type': 'rectangle', 'points': [[1, 1], ['a', 3]]},
        lm_rect('', (1, 1), (9, 9)),
        'junk',
    ]
    write_labelme(tmp_path, {'a': shapes})
    scan = scan_labelme(tmp_path)
    assert len(scan.entries[0].boxes) == 1
    assert [i.count for i in scan.issues.issues() if i.code == 'label_row_malformed'] == [4]


@pytest.mark.parametrize('ref', ['../../x.jpg', '/etc/hostname', '..\\..\\x.jpg'])
def test_escaping_image_path_falls_back_to_the_json_stem_never_outside(
    tmp_path: Path, ref: str
) -> None:
    write_image(tmp_path.parent / 'x.jpg')
    write_labelme(tmp_path / 'ds', {'a': [lm_rect('dog', (1, 1), (20, 20))]})
    path = tmp_path / 'ds/a.json'
    doc = json.loads(path.read_text())
    doc['imagePath'] = ref
    path.write_text(json.dumps(doc))
    scan = scan_labelme(tmp_path / 'ds')
    assert [e.abs_image_path.parent for e in scan.entries] == [(tmp_path / 'ds').resolve()]


def test_escaping_image_path_with_no_local_image_is_refused(tmp_path: Path) -> None:
    write_image(tmp_path.parent / 'x.jpg')
    write_labelme(tmp_path / 'ds', {'a': [lm_rect('dog', (1, 1), (20, 20))]})
    (tmp_path / 'ds/a.jpg').unlink()
    path = tmp_path / 'ds/a.json'
    doc = json.loads(path.read_text())
    doc['imagePath'] = '../x.jpg'
    path.write_text(json.dumps(doc))
    scan = scan_labelme(tmp_path / 'ds')
    assert scan.entries == []
    assert 'image_file_missing' in _codes(scan)


@pytest.mark.parametrize('body', ['not json', '[]', '{"shapes": 3}'])
def test_malformed_file_is_a_warning_and_the_image_is_not_imported(
    tmp_path: Path, body: str
) -> None:
    write_labelme(tmp_path, {'a': [], 'b': [lm_rect('dog', (1, 1), (20, 20))]})
    (tmp_path / 'a.json').write_text(body)
    scan = scan_labelme(tmp_path)
    assert 'labelme_json_invalid' in _codes(scan)
    assert not ISSUE_CATALOG['labelme_json_invalid'].blocking
    assert [e.source_stem for e in scan.entries] == ['b']


def test_oversize_file_is_blocking(tmp_path: Path, monkeypatch) -> None:
    write_labelme(tmp_path, {'a': []})
    monkeypatch.setattr(labelme, 'MAX_LABELME_JSON_BYTES', 8)
    scan = scan_labelme(tmp_path)
    assert 'dataset_file_too_large' in _codes(scan)
    assert scan.entries == []


def test_too_many_files_is_refused(tmp_path: Path, monkeypatch) -> None:
    write_labelme(tmp_path, {'a': [], 'b': []})
    monkeypatch.setenv('OP_DATASET_PREVIEW_MAX_FILES', '1')
    assert 'dataset_too_large' in _codes(scan_labelme(tmp_path))


def test_path_guard_refusal_raises(tmp_path: Path) -> None:
    write_labelme(tmp_path, {'a': []})
    with pytest.raises(DatasetPathNotAllowedError):
        scan_labelme(tmp_path, path_guard=lambda _p: False)

"""CVAT for images 1.1 XML dataset scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.annotation_fixtures import cvat_box, write_cvat
from curation.dataset_import.harness import write_image
from src.services.curation.dataset_import import cvat
from src.services.curation.dataset_import.cvat import scan_cvat
from src.services.curation.dataset_import.issues import ISSUE_CATALOG
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError
from src.services.curation.dataset_import.scan import detect_format


if TYPE_CHECKING:
    from pathlib import Path


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def test_detected_and_boxes_are_normalized_by_name(tmp_path: Path) -> None:
    write_cvat(
        tmp_path,
        {'a.jpg': [cvat_box('dog', (10, 20, 50, 80)), cvat_box('person', (60, 10, 90, 40))]},
    )
    assert detect_format(tmp_path) == 'cvat'
    scan = scan_cvat(tmp_path)
    assert scan.format == 'cvat'
    boxes = scan.entries[0].boxes
    assert [b.dataset_class for b in boxes] == ['dog', 'person']
    assert boxes[0].bbox_norm == pytest.approx((0.10, 0.20, 0.50, 0.80))
    assert scan.class_ids == {}
    assert scan.class_box_counts == {'dog': 1, 'person': 1}
    assert scan.entries[0].label_state == 'labeled'


def test_no_shapes_is_negative_and_polygon_only_is_unlabeled(tmp_path: Path) -> None:
    poly = '<polygon label="dog" points="1,1;5,5;9,1"></polygon>'
    write_cvat(tmp_path, {'neg.jpg': [], 'poly.jpg': [poly]})
    scan = scan_cvat(tmp_path)
    by = {e.source_stem: e for e in scan.entries}
    assert by['neg'].label_state == 'negative'
    assert by['poly'].label_state == 'unlabeled'
    assert 'shape_not_box_skipped' in _codes(scan)


def test_rotated_box_is_skipped(tmp_path: Path) -> None:
    write_cvat(tmp_path, {'a.jpg': [cvat_box('dog', (10, 10, 50, 50), extra=' rotation="30"')]})
    scan = scan_cvat(tmp_path)
    assert scan.entries[0].boxes == []
    assert 'shape_not_box_skipped' in _codes(scan)


def test_subset_becomes_split(tmp_path: Path) -> None:
    write_cvat(
        tmp_path,
        {'a.jpg': [cvat_box('dog', (1, 1, 20, 20))], 'b.jpg': [cvat_box('dog', (1, 1, 20, 20))]},
        subsets={'a.jpg': 'Train', 'b.jpg': 'Validation'},
    )
    by = {e.source_stem: e.split for e in scan_cvat(tmp_path).entries}
    assert by == {'a': 'train', 'b': 'val'}


def test_missing_size_is_read_from_the_image(tmp_path: Path) -> None:
    write_cvat(
        tmp_path, {'a.jpg': [cvat_box('dog', (20, 0, 100, 50))]}, size=(200, 100), xml_size=None
    )
    scan = scan_cvat(tmp_path)
    assert scan.entries[0].boxes[0].bbox_norm == pytest.approx((0.10, 0.0, 0.50, 0.50))
    assert scan.entries[0].declared_size is None


def test_wrong_declared_size_skips_the_image(tmp_path: Path) -> None:
    write_cvat(tmp_path, {'a.jpg': [cvat_box('dog', (1, 1, 20, 20))]}, xml_size=(640, 480))
    scan = scan_cvat(tmp_path)
    assert scan.entries == []
    assert 'image_size_mismatch' in _codes(scan)


@pytest.mark.parametrize(
    ('box', 'code'),
    [((1, 1, 150, 50), 'label_coords_out_of_range'), ((11, 5, 10, 50), 'label_box_degenerate')],
)
def test_bad_boxes_are_issues(tmp_path: Path, box, code: str) -> None:
    write_cvat(tmp_path, {'a.jpg': [cvat_box('dog', box)]})
    scan = scan_cvat(tmp_path)
    assert code in _codes(scan)
    assert scan.entries[0].boxes == []


def test_non_numeric_coordinate_and_empty_label_are_malformed(tmp_path: Path) -> None:
    write_cvat(tmp_path, {'a.jpg': [cvat_box('dog', (1, 1, 20, 20)), cvat_box('', (1, 1, 9, 9))]})
    xml = tmp_path / 'annotations.xml'
    xml.write_text(xml.read_text().replace('xtl="1"', 'xtl="abc"', 1))
    scan = scan_cvat(tmp_path)
    assert scan.entries[0].boxes == []
    assert [i.count for i in scan.issues.issues() if i.code == 'label_row_malformed'] == [2]


@pytest.mark.parametrize(
    'body', ['not xml at all', '<foo/>', '<!DOCTYPE a [<!ENTITY x "y">]><annotations/>']
)
def test_malformed_file_is_a_blocking_issue(tmp_path: Path, body: str) -> None:
    write_cvat(tmp_path, {'a.jpg': []})
    (tmp_path / 'annotations.xml').write_text(body)
    scan = scan_cvat(tmp_path)
    assert 'cvat_xml_invalid' in _codes(scan)
    assert ISSUE_CATALOG['cvat_xml_invalid'].blocking
    assert scan.entries == []


@pytest.mark.parametrize('ref', ['../../x.jpg', '/etc/hostname', '..\\..\\x.jpg'])
def test_escaping_image_name_is_refused(tmp_path: Path, ref: str) -> None:
    write_image(tmp_path.parent / 'x.jpg')
    write_cvat(tmp_path / 'ds', {'a.jpg': [cvat_box('dog', (1, 1, 20, 20))]})
    xml = tmp_path / 'ds/annotations.xml'
    xml.write_text(xml.read_text().replace('name="a.jpg"', f'name="{ref}"'))
    scan = scan_cvat(tmp_path / 'ds')
    assert scan.entries == []
    assert 'image_file_missing' in _codes(scan)


def test_oversize_file_is_blocking(tmp_path: Path, monkeypatch) -> None:
    write_cvat(tmp_path, {'a.jpg': []})
    monkeypatch.setattr(cvat, 'MAX_CVAT_XML_BYTES', 64)
    scan = scan_cvat(tmp_path)
    assert 'dataset_file_too_large' in _codes(scan)
    assert scan.entries == []


def test_too_many_images_is_refused(tmp_path: Path, monkeypatch) -> None:
    write_cvat(tmp_path, {'a.jpg': [], 'b.jpg': []})
    monkeypatch.setenv('OP_DATASET_PREVIEW_MAX_FILES', '1')
    assert 'dataset_too_large' in _codes(scan_cvat(tmp_path))


def test_path_guard_refusal_raises(tmp_path: Path) -> None:
    write_cvat(tmp_path, {'a.jpg': []})
    with pytest.raises(DatasetPathNotAllowedError):
        scan_cvat(tmp_path, path_guard=lambda _p: False)

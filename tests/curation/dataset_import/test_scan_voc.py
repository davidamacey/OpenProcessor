"""Pascal VOC dataset scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import pytest

from curation.dataset_import.harness import write_image
from curation.dataset_import.native_fixtures import voc_object, write_voc
from src.schemas.curation.dataset_issue import DatasetIssueCode
from src.services.curation.dataset_import import voc
from src.services.curation.dataset_import.issues import ISSUE_CATALOG
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError
from src.services.curation.dataset_import.voc import scan_voc


if TYPE_CHECKING:
    from pathlib import Path


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def test_catalog_and_wire_issue_codes_match() -> None:
    assert set(ISSUE_CATALOG) == set(get_args(DatasetIssueCode))


def test_pixels_become_normalized_one_based(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (11, 21, 50, 80))]})
    scan = scan_voc(tmp_path)
    assert scan.format == 'voc'
    assert scan.entries[0].boxes[0].bbox_norm == pytest.approx((0.10, 0.20, 0.50, 0.80))


def test_a_zero_based_xmin_is_absorbed_silently(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (0, 0, 50, 50))]})
    scan = scan_voc(tmp_path)
    assert scan.entries[0].boxes[0].bbox_norm[:2] == (0.0, 0.0)
    assert 'label_coords_clamped' not in _codes(scan)


def test_classes_are_names_and_there_are_no_indexes(tmp_path: Path) -> None:
    write_voc(
        tmp_path,
        {'a': [voc_object('dog', (1, 1, 20, 20)), voc_object('person', (30, 30, 60, 60))]},
    )
    scan = scan_voc(tmp_path)
    assert [b.dataset_class for b in scan.entries[0].boxes] == ['dog', 'person']
    assert scan.class_ids == {}
    assert scan.class_box_counts == {'dog': 1, 'person': 1}


def test_difficult_is_skipped_and_truncated_is_kept(tmp_path: Path) -> None:
    write_voc(
        tmp_path,
        {
            'hard': [voc_object('dog', (1, 1, 20, 20), difficult=1)],
            'cut': [voc_object('dog', (1, 1, 20, 20), truncated=1)],
        },
    )
    scan = scan_voc(tmp_path)
    by = {e.source_stem: e for e in scan.entries}
    assert by['hard'].label_state == 'unlabeled'
    assert by['hard'].boxes == []
    assert by['cut'].label_state == 'labeled'
    assert 'voc_difficult_skipped' in _codes(scan)


def test_no_object_is_negative(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': []})
    assert scan_voc(tmp_path).entries[0].label_state == 'negative'


def test_part_sub_boxes_are_not_objects(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': []})
    xml = tmp_path / 'Annotations/a.xml'
    part = (
        voc_object('head', (1, 1, 9, 9))
        .replace('<object>', '<part>')
        .replace('</object>', '</part>')
    )
    xml.write_text(
        xml.read_text().replace(
            '</annotation>', f'<object><name>person</name>{part}</object></annotation>'
        )
    )
    scan = scan_voc(tmp_path)
    assert scan.entries[0].boxes == []
    assert 'label_row_malformed' in _codes(scan)


def test_missing_size_is_read_from_the_image(tmp_path: Path) -> None:
    write_voc(
        tmp_path, {'a': [voc_object('dog', (21, 1, 100, 50))]}, size=(200, 100), xml_size=None
    )
    scan = scan_voc(tmp_path)
    assert 'voc_size_missing' in _codes(scan)
    assert scan.entries[0].boxes[0].bbox_norm == pytest.approx((0.10, 0.0, 0.50, 0.50))
    assert scan.entries[0].declared_size is None


def test_a_wrong_declared_size_skips_the_image(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (1, 1, 20, 20))]}, xml_size=(640, 480))
    scan = scan_voc(tmp_path)
    assert scan.entries == []
    assert 'image_size_mismatch' in _codes(scan)


def test_declared_size_is_carried_beyond_the_preview_checks(tmp_path: Path, monkeypatch) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (1, 1, 20, 20))]}, xml_size=(640, 480))
    monkeypatch.setattr(voc, 'PREVIEW_SIZE_CHECKS', 0)
    scan = scan_voc(tmp_path)
    assert scan.entries[0].declared_size == (640, 480)


@pytest.mark.parametrize(
    ('box', 'code'),
    [((1, 1, 150, 50), 'label_coords_out_of_range'), ((11, 5, 10, 50), 'label_box_degenerate')],
)
def test_bad_boxes_are_issues(tmp_path: Path, box: tuple[int, int, int, int], code: str) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', box)]})
    scan = scan_voc(tmp_path)
    assert code in _codes(scan)
    assert scan.entries[0].boxes == []


def test_non_numeric_coordinate_and_empty_name_are_malformed(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (1, 1, 20, 20)), voc_object('', (1, 1, 9, 9))]})
    xml = tmp_path / 'Annotations/a.xml'
    xml.write_text(xml.read_text().replace('<xmin>1</xmin>', '<xmin>abc</xmin>', 1))
    scan = scan_voc(tmp_path)
    assert scan.entries[0].boxes == []
    assert [i.count for i in scan.issues.issues() if i.code == 'label_row_malformed'] == [2]


def test_filename_falls_back_to_the_xml_stem(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (1, 1, 20, 20))]})
    xml = tmp_path / 'Annotations/a.xml'
    xml.write_text(xml.read_text().replace('<filename>a.jpg', '<filename>wrong.jpg'))
    scan = scan_voc(tmp_path)
    assert [e.source_stem for e in scan.entries] == ['a']


def test_missing_image_and_escaping_filename(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (1, 1, 20, 20))]})
    (tmp_path / 'JPEGImages/a.jpg').unlink()
    scan = scan_voc(tmp_path)
    assert scan.entries == []
    assert 'image_file_missing' in _codes(scan)

    outside = tmp_path.parent / 'x.jpg'
    write_image(outside)
    write_voc(tmp_path / 'ds2', {'b': [voc_object('dog', (1, 1, 20, 20))]})
    (tmp_path / 'ds2/JPEGImages/b.jpg').unlink()
    xml = tmp_path / 'ds2/Annotations/b.xml'
    xml.write_text(xml.read_text().replace('<filename>b.jpg', '<filename>../../x.jpg'))
    scan2 = scan_voc(tmp_path / 'ds2')
    assert scan2.entries == []
    assert 'image_file_missing' in _codes(scan2)


def test_splits_come_from_imagesets(tmp_path: Path) -> None:
    write_voc(
        tmp_path,
        {name: [voc_object('dog', (1, 1, 20, 20))] for name in ('a', 'b', 'c')},
        splits={'train': ['a'], 'val': ['b']},
    )
    by = {e.source_stem: e.split for e in scan_voc(tmp_path).entries}
    assert by == {'a': 'train', 'b': 'val', 'c': None}


def test_oversize_xml_is_blocking(tmp_path: Path, monkeypatch) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (1, 1, 20, 20))]})
    monkeypatch.setattr(voc, 'MAX_VOC_XML_BYTES', 64)
    scan = scan_voc(tmp_path)
    assert 'dataset_file_too_large' in _codes(scan)
    assert ISSUE_CATALOG['dataset_file_too_large'].blocking
    assert scan.entries == []


def test_too_many_files_is_refused(tmp_path: Path, monkeypatch) -> None:
    write_voc(tmp_path, {s: [voc_object('dog', (1, 1, 20, 20))] for s in ('a', 'b')})
    monkeypatch.setenv('OP_DATASET_PREVIEW_MAX_FILES', '1')
    scan = scan_voc(tmp_path)
    assert 'dataset_too_large' in _codes(scan)
    assert scan.entries == []


def test_wrong_root_tag_is_invalid(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': [voc_object('dog', (1, 1, 20, 20))]})
    (tmp_path / 'Annotations/a.xml').write_text('<foo/>')
    scan = scan_voc(tmp_path)
    assert 'voc_xml_invalid' in _codes(scan)
    assert scan.entries == []


def test_path_guard_refusal_raises(tmp_path: Path) -> None:
    write_voc(tmp_path, {'a': []})
    with pytest.raises(DatasetPathNotAllowedError):
        scan_voc(tmp_path, path_guard=lambda _p: False)

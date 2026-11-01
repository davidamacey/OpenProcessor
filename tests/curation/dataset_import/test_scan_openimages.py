"""Open Images CSV dataset scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.harness import write_image
from curation.dataset_import.native_fixtures import (
    CAR,
    DOG,
    OI_CLASSES,
    OI_CLASSES_V7,
    OI_HEADER,
    oi_row,
    write_openimages,
)
from src.services.curation.dataset_import import openimages
from src.services.curation.dataset_import.issues import ISSUE_CATALOG
from src.services.curation.dataset_import.openimages import scan_openimages
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError


if TYPE_CHECKING:
    from pathlib import Path

A = '000002b66c9c498e'
B = '0000048549557964'


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def _issue(scan, code: str):
    return next(i for i in scan.issues.issues() if i.code == code)


def test_x_x_y_y_column_order(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)]})
    scan = scan_openimages(tmp_path)
    assert scan.format == 'openimages'
    assert scan.entries[0].boxes[0].bbox_norm == pytest.approx((0.1, 0.2, 0.5, 0.8))


def test_mid_becomes_the_display_name_and_never_leaks(tmp_path: Path) -> None:
    write_openimages(
        tmp_path,
        {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8), oi_row(A, CAR, 0.5, 0.9, 0.5, 0.9)]},
    )
    scan = scan_openimages(tmp_path)
    names = [b.dataset_class for b in scan.entries[0].boxes]
    assert names == ['Dog', 'Car']
    assert scan.class_ids == {}
    assert not [k for k in scan.class_box_counts if k.startswith('/m/')]


def test_a_v7_header_row_is_not_a_class(tmp_path: Path) -> None:
    write_openimages(
        tmp_path, {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)]}, classes=OI_CLASSES_V7
    )
    scan = scan_openimages(tmp_path)
    assert set(scan.class_box_counts) == {'Dog'}
    assert 'label_row_malformed' not in _codes(scan)


def test_unknown_mid_is_skipped(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': [oi_row(A, '/m/zzz', 0.1, 0.5, 0.2, 0.8)]})
    scan = scan_openimages(tmp_path)
    assert 'openimages_class_unknown' in _codes(scan)
    assert scan.entries[0].boxes == []


def test_group_boxes_are_skipped_and_depictions_kept(tmp_path: Path) -> None:
    write_openimages(
        tmp_path,
        {
            'train': [
                oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8, group=1),
                oi_row(B, DOG, 0.1, 0.5, 0.2, 0.8, depiction=1),
            ]
        },
    )
    scan = scan_openimages(tmp_path)
    by = {e.source_stem: e for e in scan.entries}
    assert by[A].label_state == 'unlabeled'
    assert by[A].boxes == []
    assert by[B].label_state == 'labeled'
    assert 'openimages_group_skipped' in _codes(scan)


def test_an_image_is_never_negative(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': [oi_row(A, '/m/zzz', 0.1, 0.5, 0.2, 0.8)]})
    assert scan_openimages(tmp_path).entries[0].label_state == 'unlabeled'


@pytest.mark.parametrize(
    ('row', 'code'),
    [
        (oi_row(A, DOG, 0.1, 1.0005, 0.2, 0.8), 'label_coords_clamped'),
        (oi_row(A, DOG, 0.1, 1.2, 0.2, 0.8), 'label_coords_out_of_range'),
        (oi_row(A, DOG, 0.3, 0.3, 0.2, 0.8), 'label_box_degenerate'),
        (oi_row(A, DOG, 'abc', 0.5, 0.2, 0.8), 'label_row_malformed'),
        (f'{A},xclick,{DOG}\n', 'label_row_malformed'),
        (oi_row('../x', DOG, 0.1, 0.5, 0.2, 0.8), 'label_row_malformed'),
    ],
)
def test_bad_rows_are_issues(tmp_path: Path, row: str, code: str) -> None:
    write_openimages(tmp_path, {'train': [oi_row(B, DOG, 0.1, 0.5, 0.2, 0.8), row]})
    scan = scan_openimages(tmp_path)
    assert code in _codes(scan)
    if code == 'label_row_malformed':
        assert _issue(scan, code).samples[0].line == 3
    if code == 'label_coords_clamped':
        assert scan.entries[-1].boxes[0].bbox_norm[2] == 1.0


def test_missing_image_and_unreferenced_image(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)]})
    (tmp_path / f'images/train/{A}.jpg').unlink()
    write_image(tmp_path / f'images/train/{B}.jpg')
    scan = scan_openimages(tmp_path)
    assert scan.entries == []
    assert {'image_file_missing', 'openimages_image_unreferenced'} <= _codes(scan)


def test_split_comes_from_the_file_name_and_conflicts_are_reported(tmp_path: Path) -> None:
    write_openimages(
        tmp_path,
        {
            'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)],
            'validation': [oi_row(A, CAR, 0.1, 0.5, 0.2, 0.8), oi_row(B, CAR, 0.1, 0.5, 0.2, 0.8)],
        },
    )
    scan = scan_openimages(tmp_path)
    by = {e.source_stem: e for e in scan.entries}
    assert by[A].split == 'train'
    assert by[B].split == 'val'
    assert [b.dataset_class for b in by[A].boxes] == ['Dog']
    assert 'openimages_image_split_conflict' in _codes(scan)


@pytest.mark.parametrize('layout', ['images/<split>', '<split>', 'images', 'root'])
def test_each_image_layout_resolves(tmp_path: Path, layout: str) -> None:
    write_openimages(tmp_path, {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)]}, layout=layout)
    scan = scan_openimages(tmp_path)
    assert [e.source_stem for e in scan.entries] == [A]


def test_a_boxes_header_without_a_column_is_blocking(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': []})
    (tmp_path / 'train-annotations-bbox.csv').write_text('ImageID,LabelName,XMin,YMin,YMax\n')
    scan = scan_openimages(tmp_path)
    assert 'openimages_csv_invalid' in _codes(scan)
    assert ISSUE_CATALOG['openimages_csv_invalid'].blocking


def test_two_descriptions_files_are_blocking(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)]})
    (tmp_path / 'oidv7-class-descriptions-boxable.csv').write_text(OI_CLASSES)
    scan = scan_openimages(tmp_path)
    assert 'openimages_csv_invalid' in _codes(scan)
    assert scan.entries == []


def test_one_mid_with_two_names_is_blocking(tmp_path: Path) -> None:
    write_openimages(
        tmp_path,
        {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)]},
        classes=OI_CLASSES + f'{DOG},Hound\n',
    )
    assert 'openimages_csv_invalid' in _codes(scan_openimages(tmp_path))


def test_two_referenced_mids_with_one_name_are_blocking(tmp_path: Path) -> None:
    classes = '/m/a,Car\n/m/b,Car\n'
    write_openimages(
        tmp_path,
        {'train': [oi_row(A, '/m/a', 0.1, 0.5, 0.2, 0.8), oi_row(B, '/m/b', 0.1, 0.5, 0.2, 0.8)]},
        classes=classes,
    )
    scan = scan_openimages(tmp_path)
    assert 'openimages_class_duplicate_name' in _codes(scan)
    assert ISSUE_CATALOG['openimages_class_duplicate_name'].blocking


def test_a_collision_among_unused_mids_does_not_block(tmp_path: Path) -> None:
    classes = '/m/a,Car\n/m/b,Car\n'
    write_openimages(tmp_path, {'train': [oi_row(A, '/m/a', 0.1, 0.5, 0.2, 0.8)]}, classes=classes)
    assert 'openimages_class_duplicate_name' not in _codes(scan_openimages(tmp_path))


def test_an_oversize_boxes_file_is_blocking(tmp_path: Path, monkeypatch) -> None:
    write_openimages(tmp_path, {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)]})
    monkeypatch.setattr(openimages, 'MAX_OPENIMAGES_CSV_BYTES', 64)
    scan = scan_openimages(tmp_path)
    assert 'dataset_file_too_large' in _codes(scan)
    assert scan.entries == []


def test_too_many_images_is_refused(tmp_path: Path, monkeypatch) -> None:
    write_openimages(
        tmp_path,
        {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8), oi_row(B, DOG, 0.1, 0.5, 0.2, 0.8)]},
    )
    monkeypatch.setenv('OP_DATASET_PREVIEW_MAX_FILES', '1')
    scan = scan_openimages(tmp_path)
    assert 'dataset_too_large' in _codes(scan)
    assert scan.entries == []


def test_a_csv_that_makes_the_parser_raise_is_an_issue(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': []})
    (tmp_path / 'train-annotations-bbox.csv').write_text(OI_HEADER + f'{A},xclick,"{DOG},1\n')
    scan = scan_openimages(tmp_path)
    assert 'openimages_csv_invalid' in _codes(scan)
    assert ISSUE_CATALOG['openimages_csv_invalid'].blocking


def test_a_bad_utf8_boxes_file_is_an_issue(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': []})
    (tmp_path / 'train-annotations-bbox.csv').write_bytes(OI_HEADER.encode() + b'\xff\xfe\n')
    assert 'openimages_csv_invalid' in _codes(scan_openimages(tmp_path))


def test_path_guard_refusal_raises(tmp_path: Path) -> None:
    write_openimages(tmp_path, {'train': [oi_row(A, DOG, 0.1, 0.5, 0.2, 0.8)]})
    with pytest.raises(DatasetPathNotAllowedError):
        scan_openimages(tmp_path, path_guard=lambda _p: False)

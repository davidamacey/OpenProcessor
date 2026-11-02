"""KITTI object-detection (label_2) dataset scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.annotation_fixtures import kitti_line, write_kitti
from curation.dataset_import.harness import write_image
from src.services.curation.dataset_import import kitti
from src.services.curation.dataset_import.kitti import scan_kitti
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError
from src.services.curation.dataset_import.scan import detect_format


if TYPE_CHECKING:
    from pathlib import Path


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def test_detected_and_pixels_normalized_by_name(tmp_path: Path) -> None:
    lines = [kitti_line('Car', (20, 10, 100, 60)), kitti_line('Person_sitting', (120, 20, 180, 90))]
    write_kitti(tmp_path, {'000001': lines}, size=(200, 100))
    assert detect_format(tmp_path) == 'kitti'
    scan = scan_kitti(tmp_path)
    assert scan.format == 'kitti'
    entry = scan.entries[0]
    assert entry.label_state == 'labeled'
    assert [b.dataset_class for b in entry.boxes] == ['Car', 'Person_sitting']
    assert entry.boxes[0].bbox_norm == pytest.approx((0.10, 0.10, 0.50, 0.60))
    assert scan.class_ids == {}
    assert scan.class_box_counts == {'Car': 1, 'Person_sitting': 1}


def test_training_subdirectory_layout(tmp_path: Path) -> None:
    write_kitti(tmp_path / 'training', {'000001': [kitti_line('Car', (20, 10, 100, 60))]})
    assert detect_format(tmp_path) == 'kitti'
    assert len(scan_kitti(tmp_path).entries) == 1


def test_empty_file_is_negative_and_dontcare_only_is_unlabeled(tmp_path: Path) -> None:
    write_kitti(
        tmp_path,
        {'neg': [], 'dc': [kitti_line('DontCare', (1, 1, 20, 20))]},
    )
    scan = scan_kitti(tmp_path)
    by = {e.source_stem: e for e in scan.entries}
    assert by['neg'].label_state == 'negative'
    assert by['dc'].label_state == 'unlabeled'
    assert by['dc'].boxes == []
    assert 'shape_not_box_skipped' in _codes(scan)


def test_a_label_file_is_required_for_an_image_to_be_part_of_the_dataset(tmp_path: Path) -> None:
    write_kitti(tmp_path, {'a': [kitti_line('Car', (1, 1, 20, 20))], 'unlabelled': None})
    assert [e.source_stem for e in scan_kitti(tmp_path).entries] == ['a']


def test_scored_prediction_lines_have_16_fields_and_are_accepted(tmp_path: Path) -> None:
    write_kitti(tmp_path, {'a': [kitti_line('Car', (1, 1, 20, 20), extra=' 0.9')]})
    assert len(scan_kitti(tmp_path).entries[0].boxes) == 1


def test_splits_come_from_imagesets(tmp_path: Path) -> None:
    write_kitti(
        tmp_path,
        {n: [kitti_line('Car', (1, 1, 20, 20))] for n in ('a', 'b', 'c')},
        splits={'train': ['a'], 'val': ['b']},
    )
    by = {e.source_stem: e.split for e in scan_kitti(tmp_path).entries}
    assert by == {'a': 'train', 'b': 'val', 'c': None}


@pytest.mark.parametrize(
    ('box', 'code'),
    [((1, 1, 450, 50), 'label_coords_out_of_range'), ((30, 5, 10, 50), 'label_box_degenerate')],
)
def test_bad_boxes_are_issues(tmp_path: Path, box, code: str) -> None:
    write_kitti(tmp_path, {'a': [kitti_line('Car', box)]})
    scan = scan_kitti(tmp_path)
    assert code in _codes(scan)
    assert scan.entries[0].boxes == []


@pytest.mark.parametrize(
    'line',
    ['Car 0 0 0 1 2 3', 'Car 0 0 0 a b c d 1 1 1 1 1 1 1', 'Car 0 0 0 nan 1 9 9 1 1 1 1 1 1 1'],
)
def test_malformed_lines_are_skipped_with_a_line_number(tmp_path: Path, line: str) -> None:
    write_kitti(tmp_path, {'a': [kitti_line('Car', (1, 1, 20, 20)), line]})
    scan = scan_kitti(tmp_path)
    assert len(scan.entries[0].boxes) == 1
    issue = next(
        i
        for i in scan.issues.issues()
        if i.code in ('label_row_malformed', 'label_coords_out_of_range')
    )
    assert issue.samples[0].line == 2


def test_missing_image_is_reported(tmp_path: Path) -> None:
    write_kitti(tmp_path, {'a': [kitti_line('Car', (1, 1, 20, 20))]})
    (tmp_path / 'image_2/a.png').unlink()
    scan = scan_kitti(tmp_path)
    assert scan.entries == []
    assert 'image_file_missing' in _codes(scan)


def test_a_label_file_name_cannot_reach_outside_the_image_dir(tmp_path: Path) -> None:
    write_image(tmp_path.parent / 'x.png')
    write_kitti(tmp_path / 'ds', {'a': [kitti_line('Car', (1, 1, 20, 20))]})
    (tmp_path / 'ds/image_2/a.png').unlink()
    (tmp_path / 'ds/image_2/a.png').symlink_to(tmp_path.parent / 'x.png')
    scan = scan_kitti(tmp_path / 'ds')
    assert scan.entries == []
    assert 'image_file_missing' in _codes(scan)


def test_oversize_label_file_is_blocking(tmp_path: Path, monkeypatch) -> None:
    write_kitti(tmp_path, {'a': [kitti_line('Car', (1, 1, 20, 20))]})
    monkeypatch.setattr(kitti, 'MAX_LABEL_FILE_BYTES', 16)
    scan = scan_kitti(tmp_path)
    assert 'dataset_file_too_large' in _codes(scan)
    assert scan.entries == []


def test_too_many_files_is_refused(tmp_path: Path, monkeypatch) -> None:
    write_kitti(tmp_path, {'a': [], 'b': []})
    monkeypatch.setenv('OP_DATASET_PREVIEW_MAX_FILES', '1')
    assert 'dataset_too_large' in _codes(scan_kitti(tmp_path))


def test_path_guard_refusal_raises(tmp_path: Path) -> None:
    write_kitti(tmp_path, {'a': []})
    with pytest.raises(DatasetPathNotAllowedError):
        scan_kitti(tmp_path, path_guard=lambda _p: False)

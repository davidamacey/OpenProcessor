"""Label Studio JSON export (rectanglelabels) dataset scanning."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.annotation_fixtures import ls_rect, ls_task, write_labelstudio
from curation.dataset_import.harness import write_image
from src.services.curation.dataset_import import labelstudio
from src.services.curation.dataset_import.issues import ISSUE_CATALOG
from src.services.curation.dataset_import.labelstudio import scan_labelstudio
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError
from src.services.curation.dataset_import.scan import detect_format


if TYPE_CHECKING:
    from pathlib import Path

IMG = {'a.jpg': (100, 100)}


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def test_detected_and_percent_boxes_become_normalized_names(tmp_path: Path) -> None:
    tasks = [ls_task('a.jpg', [ls_rect('Dog', 10, 20, 40, 60), ls_rect('Person', 60, 10, 30, 30)])]
    write_labelstudio(tmp_path, tasks, images=IMG)
    assert detect_format(tmp_path) == 'labelstudio'
    scan = scan_labelstudio(tmp_path)
    assert scan.format == 'labelstudio'
    boxes = scan.entries[0].boxes
    assert [b.dataset_class for b in boxes] == ['Dog', 'Person']
    assert boxes[0].bbox_norm == pytest.approx((0.10, 0.20, 0.50, 0.80))
    assert scan.class_ids == {}
    assert scan.entries[0].declared_size == (100, 100)


def test_empty_annotation_is_negative_and_no_annotation_is_unlabeled(tmp_path: Path) -> None:
    tasks = [ls_task('a.jpg', []), ls_task('b.jpg', None)]
    write_labelstudio(tmp_path, tasks, images={'a.jpg': (100, 100), 'b.jpg': (100, 100)})
    by = {e.source_stem: e.label_state for e in scan_labelstudio(tmp_path).entries}
    assert by == {'a': 'negative', 'b': 'unlabeled'}


def test_cancelled_annotations_do_not_count(tmp_path: Path) -> None:
    tasks = [ls_task('a.jpg', [ls_rect('Dog', 1, 1, 20, 20)], was_cancelled=True)]
    write_labelstudio(tmp_path, tasks, images=IMG)
    entry = scan_labelstudio(tmp_path).entries[0]
    assert (entry.label_state, entry.boxes) == ('unlabeled', [])


def test_several_annotations_are_refused_not_guessed(tmp_path: Path) -> None:
    task = ls_task('a.jpg', [ls_rect('Dog', 1, 1, 20, 20)])
    task['annotations'].append(dict(task['annotations'][0], id=2))
    write_labelstudio(tmp_path, [task], images=IMG)
    scan = scan_labelstudio(tmp_path)
    assert scan.entries == []
    assert 'labelstudio_task_skipped' in _codes(scan)


def test_rotated_and_non_rectangle_results(tmp_path: Path) -> None:
    poly = {'type': 'polygonlabels', 'value': {'points': [[1, 1], [5, 5], [9, 1]]}}
    choice = {'type': 'choices', 'value': {'choices': ['cat']}}
    tasks = [ls_task('a.jpg', [ls_rect('Dog', 10, 10, 20, 20, rotation=15), poly, choice])]
    write_labelstudio(tmp_path, tasks, images=IMG)
    scan = scan_labelstudio(tmp_path)
    assert scan.entries[0].boxes == []
    assert scan.entries[0].label_state == 'unlabeled'
    assert [i.count for i in scan.issues.issues() if i.code == 'shape_not_box_skipped'] == [2]


def test_a_result_with_several_labels_is_malformed(tmp_path: Path) -> None:
    rect = ls_rect('Dog', 10, 10, 20, 20)
    rect['value']['rectanglelabels'] = ['Dog', 'Cat']
    write_labelstudio(tmp_path, [ls_task('a.jpg', [rect])], images=IMG)
    scan = scan_labelstudio(tmp_path)
    assert scan.entries[0].boxes == []
    assert 'label_row_malformed' in _codes(scan)


@pytest.mark.parametrize(
    ('rect', 'code'),
    [
        (('Dog', 90, 10, 30, 20), 'label_coords_out_of_range'),
        (('Dog', 10, 10, 0, 20), 'label_box_degenerate'),
    ],
)
def test_bad_boxes_are_issues(tmp_path: Path, rect, code: str) -> None:
    write_labelstudio(tmp_path, [ls_task('a.jpg', [ls_rect(*rect)])], images=IMG)
    scan = scan_labelstudio(tmp_path)
    assert code in _codes(scan)
    assert scan.entries[0].boxes == []


def test_local_files_and_upload_prefixes_resolve(tmp_path: Path) -> None:
    tasks = [
        ls_task('/data/local-files/?d=images/a.jpg', [ls_rect('Dog', 1, 1, 20, 20)]),
        ls_task('/data/upload/3/b.jpg', [ls_rect('Dog', 1, 1, 20, 20, size=(101, 100))]),
    ]
    write_labelstudio(tmp_path, tasks, images={'a.jpg': (100, 100), 'b.jpg': (100, 100)})
    assert sorted(e.source_stem for e in scan_labelstudio(tmp_path).entries) == ['a', 'b']


@pytest.mark.parametrize(
    'ref',
    [
        '../../x.jpg',
        '/etc/hostname',
        '/data/local-files/?d=../../x.jpg',
        '/data/upload/1/../../x.jpg',
        'https://example.org/a.jpg',
        's3://bucket/a.jpg',
    ],
)
def test_escaping_or_remote_image_refs_are_refused(tmp_path: Path, ref: str) -> None:
    write_image(tmp_path.parent / 'x.jpg')
    write_labelstudio(tmp_path / 'ds', [ls_task(ref, [ls_rect('Dog', 1, 1, 20, 20)])])
    scan = scan_labelstudio(tmp_path / 'ds')
    assert scan.entries == []
    assert 'image_file_missing' in _codes(scan)


@pytest.mark.parametrize('body', ['not json', '{"a": 1}', '[1, 2]', '[{"data": 5}]'])
def test_malformed_file_is_a_blocking_issue(tmp_path: Path, body: str) -> None:
    write_labelstudio(tmp_path, [ls_task('a.jpg', [])], images=IMG)
    (tmp_path / 'project-1-export.json').write_text(body)
    scan = scan_labelstudio(tmp_path)
    assert 'labelstudio_json_invalid' in _codes(scan)
    assert ISSUE_CATALOG['labelstudio_json_invalid'].blocking
    assert scan.entries == []


def test_deeply_nested_json_is_invalid_not_a_crash(tmp_path: Path) -> None:
    write_labelstudio(tmp_path, [ls_task('a.jpg', [])], images=IMG)
    (tmp_path / 'project-1-export.json').write_text('[' * 100000)
    assert 'labelstudio_json_invalid' in _codes(scan_labelstudio(tmp_path))


def test_oversize_file_is_blocking(tmp_path: Path, monkeypatch) -> None:
    write_labelstudio(tmp_path, [ls_task('a.jpg', [])], images=IMG)
    monkeypatch.setattr(labelstudio, 'MAX_LABELSTUDIO_JSON_BYTES', 16)
    scan = scan_labelstudio(tmp_path)
    assert 'dataset_file_too_large' in _codes(scan)
    assert scan.entries == []


def test_too_many_tasks_is_refused(tmp_path: Path, monkeypatch) -> None:
    tasks = [ls_task('a.jpg', []), ls_task('b.jpg', [])]
    write_labelstudio(tmp_path, tasks, images={'a.jpg': (100, 100), 'b.jpg': (100, 100)})
    monkeypatch.setenv('OP_DATASET_PREVIEW_MAX_FILES', '1')
    assert 'dataset_too_large' in _codes(scan_labelstudio(tmp_path))


def test_path_guard_refusal_raises(tmp_path: Path) -> None:
    write_labelstudio(tmp_path, [ls_task('a.jpg', [])], images=IMG)
    with pytest.raises(DatasetPathNotAllowedError):
        scan_labelstudio(tmp_path, path_guard=lambda _p: False)


def test_a_non_label_studio_json_is_not_detected(tmp_path: Path) -> None:
    (tmp_path / 'other.json').write_text(json.dumps([{'x': 1}]))
    (tmp_path / 'images').mkdir()
    assert detect_format(tmp_path) == 'yolo'

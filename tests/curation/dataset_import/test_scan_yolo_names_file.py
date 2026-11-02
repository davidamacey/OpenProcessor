"""YOLO datasets whose class names come from a names file (``classes.txt`` or
``obj.names``) instead of a ``data.yaml``. The names file is the only source
of class identity; with none, the dataset is refused."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.harness import write_image
from src.services.curation.dataset_import import yolo
from src.services.curation.dataset_import.issues import ISSUE_CATALOG
from src.services.curation.dataset_import.options import DatasetSource
from src.services.curation.dataset_import.scan import (
    FormatUndetectedError,
    detect_format,
    scan_dataset,
)
from src.services.curation.dataset_import.yolo import scan_yolo


if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, names: str, images: dict[str, list[str]], *, split: str = '') -> None:
    (root / 'classes.txt').write_text(names)
    folder = f'{split}/' if split else ''
    (root / f'labels/{folder}').mkdir(parents=True, exist_ok=True)
    for i, (stem, rows) in enumerate(images.items()):
        write_image(root / f'images/{folder}{stem}.jpg', color=['red', 'green', 'blue'][i % 3])
        (root / f'labels/{folder}{stem}.txt').write_text(''.join(f'{r}\n' for r in rows))


def _codes(scan) -> set[str]:
    return {i.code for i in scan.issues.issues()}


def test_names_come_from_classes_txt_by_line(tmp_path: Path) -> None:
    _write(
        tmp_path,
        'car\ntruck\nbus\n',
        {'a': ['2 0.5 0.5 0.2 0.2', '0 0.25 0.25 0.1 0.1'], 'neg': []},
    )
    assert detect_format(tmp_path) == 'yolo'
    scan = scan_dataset(
        DatasetSource.model_validate({'path': str(tmp_path)}), path_guard=lambda _p: True
    )
    by = {e.source_stem: e for e in scan.entries}
    assert [b.dataset_class for b in by['a'].boxes] == ['bus', 'car']
    assert by['a'].boxes[0].bbox_norm == pytest.approx((0.4, 0.4, 0.6, 0.6))
    assert by['neg'].label_state == 'negative'
    assert by['a'].split is None
    assert scan.class_ids == {'car': 0, 'truck': 1, 'bus': 2}


def test_obj_names_and_split_directories(tmp_path: Path) -> None:
    (tmp_path / 'obj.names').write_text('car\ntruck')
    for split, stem in (('train', 'a'), ('valid', 'b')):
        write_image(
            tmp_path / f'images/{split}/{stem}.jpg', color='green' if stem == 'b' else 'red'
        )
        (tmp_path / f'labels/{split}').mkdir(parents=True, exist_ok=True)
        (tmp_path / f'labels/{split}/{stem}.txt').write_text('1 0.5 0.5 0.2 0.2\n')
    by = {e.source_stem: (e.split, e.boxes[0].dataset_class) for e in scan_yolo(tmp_path).entries}
    assert by == {'a': ('train', 'truck'), 'b': ('val', 'truck')}


def test_images_without_any_names_source_are_refused(tmp_path: Path) -> None:
    write_image(tmp_path / 'images/a.jpg')
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'labels/a.txt').write_text('0 0.5 0.5 0.2 0.2\n')
    with pytest.raises(FormatUndetectedError):
        scan_yolo(tmp_path)
    with pytest.raises(FormatUndetectedError):
        scan_dataset(
            DatasetSource.model_validate({'path': str(tmp_path)}), path_guard=lambda _p: True
        )


def test_an_index_beyond_the_names_is_never_guessed(tmp_path: Path) -> None:
    _write(tmp_path, 'car\n', {'a': ['0 0.5 0.5 0.2 0.2', '1 0.5 0.5 0.2 0.2']})
    scan = scan_yolo(tmp_path)
    assert [b.dataset_class for b in scan.entries[0].boxes] == ['car']
    assert 'label_class_out_of_range' in _codes(scan)


@pytest.mark.parametrize('body', ['', '\n\n', 'car\n\ntruck\n', 'car\ntruck\ncar\n'])
def test_empty_blank_or_duplicate_names_block_the_import(tmp_path: Path, body: str) -> None:
    _write(tmp_path, body, {'a': ['0 0.5 0.5 0.2 0.2']})
    scan = scan_yolo(tmp_path)
    assert 'yolo_names_file_invalid' in _codes(scan)
    assert ISSUE_CATALOG['yolo_names_file_invalid'].blocking
    assert scan.entries == []


def test_oversize_names_file_is_blocking(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, 'car\ntruck\n', {'a': ['0 0.5 0.5 0.2 0.2']})
    monkeypatch.setattr(yolo, 'MAX_YAML_BYTES', 4)
    scan = scan_yolo(tmp_path)
    assert 'yolo_names_file_invalid' in _codes(scan)
    assert scan.entries == []


def test_a_names_file_outside_the_allowed_roots_is_not_read(tmp_path: Path) -> None:
    _write(tmp_path, 'car\n', {'a': ['0 0.5 0.5 0.2 0.2']})
    scan = scan_yolo(tmp_path, path_guard=lambda p: p.name != 'classes.txt')
    assert 'dataset_path_not_allowed' in _codes(scan)
    assert scan.entries == []


def test_data_yaml_still_wins_over_a_names_file(tmp_path: Path) -> None:
    _write(tmp_path, 'wrong\n', {'a': ['0 0.5 0.5 0.2 0.2']})
    (tmp_path / 'data.yaml').write_text('train: images\nnames:\n  0: right\n')
    assert scan_yolo(tmp_path).entries[0].boxes[0].dataset_class == 'right'

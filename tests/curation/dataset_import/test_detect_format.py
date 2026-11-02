"""Format detection precedence: the first matching rule wins."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.services.curation.dataset_import.options import DatasetSource
from src.services.curation.dataset_import.scan import (
    FormatUndetectedError,
    detect_format,
    scan_dataset,
)


if TYPE_CHECKING:
    from pathlib import Path

_OI = ('train-annotations-bbox.csv', 'class-descriptions-boxable.csv')

_CASES = [
    pytest.param(('Annotations/a.xml', 'JPEGImages/'), 'voc', id='voc-capitalised'),
    pytest.param(('annotations/a.xml', 'images/'), 'voc', id='voc-lowercase-with-images'),
    pytest.param(('Annotations/a.xml', 'data.yaml'), 'yolo', id='data-yaml-beats-voc'),
    pytest.param(
        ('annotations/a.json', 'annotations/b.xml', 'images/'), 'coco', id='json-beats-xml'
    ),
    pytest.param(
        ('annotations/a.json', 'annotations/b.xml', 'images/', 'labels/'),
        'yolo',
        id='labels-dir-keeps-yolo',
    ),
    pytest.param(('images/',), 'yolo', id='images-only'),
    pytest.param(_OI, 'openimages', id='oi'),
    pytest.param((*_OI, 'images/'), 'openimages', id='oi-with-images'),
    pytest.param((*_OI, 'data.yaml'), 'yolo', id='data-yaml-beats-oi'),
    pytest.param((*_OI, 'Annotations/a.xml'), 'voc', id='voc-beats-oi'),
    pytest.param(('annotations.xml', 'images/'), 'yolo', id='empty-annotations-xml-is-not-cvat'),
    pytest.param(('annotations.xml', 'data.yaml'), 'yolo', id='data-yaml-beats-cvat'),
    pytest.param(('label_2/a.txt', 'image_2/'), 'kitti', id='kitti'),
    pytest.param(('training/label_2/a.txt',), 'kitti', id='kitti-training-dir'),
    pytest.param(('label_2/a.txt', 'data.yaml'), 'yolo', id='data-yaml-beats-kitti'),
    pytest.param(('classes.txt', 'images/'), 'yolo', id='classes-txt-layout'),
    pytest.param(('train-annotations-bbox.csv',), None, id='oi-without-descriptions'),
    pytest.param(('Annotations/',), None, id='annotations-without-xml'),
]


def _build(root: Path, layout: tuple[str, ...]) -> None:
    for rel in layout:
        target = root / rel
        if rel.endswith('/'):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('')


@pytest.mark.parametrize(('layout', 'expected'), _CASES)
def test_detect_format_table(tmp_path: Path, layout: tuple[str, ...], expected: str | None) -> None:
    _build(tmp_path, layout)
    if expected is None:
        with pytest.raises(FormatUndetectedError):
            detect_format(tmp_path)
    else:
        assert detect_format(tmp_path) == expected


@pytest.mark.parametrize('fmt', ['voc', 'openimages'])
def test_an_explicit_format_dispatches_to_its_own_reader(tmp_path: Path, fmt: str) -> None:
    if fmt == 'voc':
        _build(tmp_path, ('Annotations/', 'JPEGImages/'))
    else:
        _build(tmp_path, _OI)
    source = DatasetSource.model_validate({'path': str(tmp_path), 'format': fmt})
    scan = scan_dataset(source, path_guard=lambda _p: True)
    assert scan.format == fmt

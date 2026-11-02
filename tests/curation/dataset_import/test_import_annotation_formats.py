"""CVAT, Label Studio, KITTI, CreateML and LabelMe imports end to end through
the runner: class identity by name, the lock rule, negatives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.annotation_fixtures import (
    cml_ann,
    cvat_box,
    kitti_line,
    lm_rect,
    ls_rect,
    ls_task,
    write_createml,
    write_cvat,
    write_kitti,
    write_labelme,
    write_labelstudio,
)
from curation.dataset_import.harness import Harness, map_all, write_image


if TYPE_CHECKING:
    from pathlib import Path


def _items_of(h: Harness, class_name: str) -> list[dict]:
    return [d for d in h.items.values() if d.get('class_name') == class_name]


@pytest.mark.asyncio
async def test_cvat_import_maps_by_name_and_keeps_negatives(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('person')
    h.registry.add_class('dog')
    root = tmp_path / 'ds'
    write_cvat(
        root,
        {
            'a.jpg': [cvat_box('dog', (1, 1, 40, 40)), cvat_box('person', (50, 50, 90, 90))],
            'neg.jpg': [],
        },
    )
    prepared = h.prepare(h.request(root))
    assert prepared.scan.format == 'cvat'
    assert h.items == {}
    store, _ = await h.run(h.request(root, map_all(h.registry, 'dog', 'person')))
    report = store.job.read()['report']
    assert store.job.read()['status'] == 'completed'
    assert (report['negatives'], report['items_created']) == (1, 2)
    dog = _items_of(h, 'dog')[0]
    person = _items_of(h, 'person')[0]
    assert (dog['class_id'], person['class_id']) == (1, 0)
    assert all(d['label_source'] == 'import' for d in h.items.values())


@pytest.mark.asyncio
async def test_cvat_lock_rule(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('dog')
    root = tmp_path / 'ds'
    write_cvat(root, {'a.jpg': [cvat_box('dog', (1, 1, 40, 40))]})
    mapping = map_all(h.registry, 'dog')
    await h.run(h.request(root, mapping))
    cid = _items_of(h, 'dog')[0]['crop_id']
    h.items[cid].update(
        {'class_id': 99, 'class_name': 'bus', 'class_source': 'human', 'label_source': 'human'}
    )
    root2 = tmp_path / 'ds2'
    write_cvat(root2, {'a.jpg': [cvat_box('dog', (1, 1, 40, 40))], 'b.jpg': []})
    (root2 / 'images/a.jpg').write_bytes((root / 'images/a.jpg').read_bytes())
    store2, _ = await h.run(h.request(root2, mapping, name='v2'))
    after = h.items[cid]
    assert (after['class_id'], after['class_name'], after['class_source']) == (99, 'bus', 'human')
    assert store2.job.read()['report']['label_conflicts_locked'] == 1


def _build_labelstudio(root: Path) -> None:
    tasks = [
        ls_task('a.jpg', [ls_rect('dog', 1, 1, 39, 39), ls_rect('person', 50, 50, 40, 40)]),
        ls_task('neg.jpg', []),
    ]
    write_labelstudio(root, tasks, images={'a.jpg': (100, 100), 'neg.jpg': (100, 100)})


def _build_kitti(root: Path) -> None:
    lines = [kitti_line('dog', (1, 1, 40, 40)), kitti_line('person', (50, 50, 90, 90))]
    write_kitti(root, {'a': lines, 'neg': []}, size=(100, 100))


def _build_createml(root: Path) -> None:
    anns = [cml_ann('dog', 20, 20, 39, 39), cml_ann('person', 70, 70, 40, 40)]
    write_createml(root, {'a.jpg': anns, 'neg.jpg': []})


def _build_labelme(root: Path) -> None:
    shapes = [lm_rect('dog', (1, 1), (40, 40)), lm_rect('person', (50, 50), (90, 90))]
    write_labelme(root, {'a': shapes, 'neg': []})


def _build_yolo_names(root: Path) -> None:
    (root / 'labels').mkdir(parents=True)
    (root / 'classes.txt').write_text('person\ndog\n')
    rows_by_stem = {'a': ['1 0.2 0.2 0.39 0.39', '0 0.7 0.7 0.4 0.4'], 'neg': []}
    for i, (stem, rows) in enumerate(rows_by_stem.items()):
        write_image(root / f'images/{stem}.jpg', size=(100 + i, 100), color='red')
        (root / f'labels/{stem}.txt').write_text(''.join(f'{r}\n' for r in rows))


_BUILDERS = {
    'labelstudio': _build_labelstudio,
    'kitti': _build_kitti,
    'createml': _build_createml,
    'labelme': _build_labelme,
    'yolo': _build_yolo_names,
}


@pytest.mark.asyncio
@pytest.mark.parametrize('fmt', list(_BUILDERS))
async def test_other_formats_import_by_name_with_negatives(
    fmt: str, tmp_path: Path, monkeypatch
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('person')
    h.registry.add_class('dog')
    root = tmp_path / 'ds'
    _BUILDERS[fmt](root)
    prepared = h.prepare(h.request(root))
    assert prepared.scan.format == fmt
    assert set(prepared.scan.class_box_counts) == {'dog', 'person'}
    assert h.items == {}
    store, _ = await h.run(h.request(root, map_all(h.registry, 'dog', 'person')))
    report = store.job.read()['report']
    assert store.job.read()['status'] == 'completed'
    assert (report['negatives'], report['items_created']) == (1, 2)
    dog, person = _items_of(h, 'dog')[0], _items_of(h, 'person')[0]
    assert (dog['class_id'], person['class_id']) == (1, 0)
    assert dog['bbox_norm'][0] < person['bbox_norm'][0]
    assert all(d['label_source'] == 'import' for d in h.items.values())

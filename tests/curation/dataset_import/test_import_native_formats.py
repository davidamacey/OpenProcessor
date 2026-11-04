"""Pascal VOC and Open Images imports end to end through the runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.harness import Harness, map_all
from curation.dataset_import.native_fixtures import (
    CAR,
    DOG,
    PERSON,
    oi_row,
    voc_object,
    write_openimages,
    write_voc,
)


if TYPE_CHECKING:
    from pathlib import Path


def _items_of(h: Harness, class_name: str) -> list[dict]:
    return [d for d in h.items.values() if d.get('class_name') == class_name]


@pytest.mark.asyncio
async def test_voc_preview_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    root = tmp_path / 'ds'
    write_voc(
        root,
        {'a': [voc_object('dog', (1, 1, 40, 40)), voc_object('person', (50, 50, 90, 90))]},
    )
    prepared = h.prepare(h.request(root))
    assert prepared.scan.format == 'voc'
    assert set(prepared.scan.class_box_counts) == {'dog', 'person'}
    assert h.items == {}
    assert h.images == {}


@pytest.mark.asyncio
async def test_voc_import_maps_by_name_not_order(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add('person')
    h.registry.add('dog')
    root = tmp_path / 'ds'
    write_voc(
        root,
        {'a': [voc_object('dog', (1, 1, 40, 40)), voc_object('person', (50, 50, 90, 90))]},
    )
    store, _ = await h.run(h.request(root, map_all(h.registry, 'dog', 'person')))
    assert store.job.read()['status'] == 'completed'
    dog = _items_of(h, 'dog')[0]
    person = _items_of(h, 'person')[0]
    assert (dog['class_name'], dog['class_id']) == ('dog', 1)
    assert (person['class_name'], person['class_id']) == ('person', 0)
    assert dog['bbox_norm'][0] < person['bbox_norm'][0]
    for doc in h.items.values():
        assert doc['class_source'] == 'external_label'
        assert doc['class_validated'] is True
        assert doc['label_source'] == 'import'


@pytest.mark.asyncio
async def test_voc_negative_and_split(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add('dog')
    root = tmp_path / 'ds'
    write_voc(
        root,
        {'pos': [voc_object('dog', (1, 1, 40, 40))], 'neg': []},
        splits={'val': ['pos', 'neg']},
    )
    store, _ = await h.run(h.request(root, map_all(h.registry, 'dog')))
    report = store.job.read()['report']
    assert (report['negatives'], report['items_created']) == (1, 1)
    assert all(d['dataset_split'] == 'val' for d in h.images.values())
    assert all(d['dataset_split'] == 'val' for d in h.items.values())


@pytest.mark.asyncio
async def test_voc_lock_rule(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add('dog')
    root = tmp_path / 'ds'
    write_voc(root, {'a': [voc_object('dog', (1, 1, 40, 40))]})
    mapping = map_all(h.registry, 'dog')
    await h.run(h.request(root, mapping))
    cid = _items_of(h, 'dog')[0]['crop_id']
    h.items[cid].update(
        {'class_id': 99, 'class_name': 'bus', 'class_source': 'human', 'label_source': 'human'}
    )
    root2 = tmp_path / 'ds2'
    write_voc(root2, {'a': [voc_object('dog', (1, 1, 40, 40))], 'b': []})
    (root2 / 'JPEGImages/a.jpg').write_bytes((root / 'JPEGImages/a.jpg').read_bytes())
    store2, _ = await h.run(h.request(root2, mapping, name='v2'))
    after = h.items[cid]
    assert (after['class_id'], after['class_name'], after['class_source']) == (99, 'bus', 'human')
    assert store2.job.read()['report']['label_conflicts_locked'] == 1


@pytest.mark.asyncio
async def test_openimages_import_maps_display_name(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add('Car')
    h.registry.add('Dog')
    root = tmp_path / 'ds'
    write_openimages(
        root,
        {
            'train': [
                oi_row('000002b66c9c498e', DOG, 0.1, 0.4, 0.1, 0.4),
                oi_row('000002b66c9c498e', CAR, 0.5, 0.9, 0.5, 0.9),
            ]
        },
    )
    store, _ = await h.run(h.request(root, map_all(h.registry, 'Dog', 'Car')))
    assert store.job.read()['status'] == 'completed'
    dog = _items_of(h, 'Dog')[0]
    car = _items_of(h, 'Car')[0]
    assert (dog['class_id'], car['class_id']) == (1, 0)
    assert not [d for d in h.items.values() if str(d['class_name']).startswith('/m/')]
    assert all(d['label_source'] == 'import' for d in h.items.values())


@pytest.mark.asyncio
async def test_openimages_preview_writes_nothing_and_lists_names(
    tmp_path: Path, monkeypatch
) -> None:
    h = Harness(tmp_path, monkeypatch)
    root = tmp_path / 'ds'
    write_openimages(root, {'validation': [oi_row('0000048549557964', PERSON, 0.1, 0.4, 0.1, 0.4)]})
    prepared = h.prepare(h.request(root))
    assert prepared.scan.format == 'openimages'
    assert set(prepared.scan.class_box_counts) == {'Person'}
    assert h.items == {}
    assert h.images == {}


@pytest.mark.asyncio
async def test_openimages_lock_rule(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add('Dog')
    root = tmp_path / 'ds'
    write_openimages(root, {'train': [oi_row('000002b66c9c498e', DOG, 0.1, 0.4, 0.1, 0.4)]})
    mapping = map_all(h.registry, 'Dog')
    await h.run(h.request(root, mapping))
    cid = _items_of(h, 'Dog')[0]['crop_id']
    h.items[cid].update(
        {'class_id': 99, 'class_name': 'bus', 'class_source': 'human', 'label_source': 'human'}
    )
    root2 = tmp_path / 'ds2'
    write_openimages(
        root2,
        {
            'train': [
                oi_row('000002b66c9c498e', DOG, 0.1, 0.4, 0.1, 0.4),
                oi_row('0000048549557964', DOG, 0.1, 0.4, 0.1, 0.4),
            ]
        },
    )
    same = root / 'images/train/000002b66c9c498e.jpg'
    (root2 / 'images/train/000002b66c9c498e.jpg').write_bytes(same.read_bytes())
    store2, _ = await h.run(h.request(root2, mapping, name='v2'))
    after = h.items[cid]
    assert (after['class_id'], after['class_name'], after['class_source']) == (99, 'bus', 'human')
    assert store2.job.read()['report']['label_conflicts_locked'] == 1

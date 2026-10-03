"""The import job end to end (W10.6): class mapping by name, ledger,
locks, regions, negatives, splits. Ports every assertion of the retired
``test_job_import.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from curation.dataset_import.harness import Harness, activate_region_profile, map_all, write_yolo
from src.config.region_fields import get_region_fields
from src.services.curation.dataset_import.mapping import ClassMappingEntry
from src.services.curation.region_boxes import read_boxes


def _items_of(h: Harness, class_name: str) -> list[dict]:
    return [d for d in h.items.values() if d.get('class_name') == class_name]


@pytest.mark.asyncio
async def test_import_maps_classes_by_name_not_index(tmp_path: Path, monkeypatch) -> None:
    """Dataset ``{0: truck, 1: car}`` against registry ``[car(0), truck(1)]``
    must import trucks as truck: the pre-W10 bug this wave fixes."""
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    h.registry.add_class('truck')
    root = tmp_path / 'ds'
    write_yolo(
        root,
        names=['truck', 'car'],
        images={'a': ['1 0.3 0.3 0.2 0.2', '0 0.7 0.7 0.2 0.2']},
    )
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car', 'truck')))
    assert store.job.read()['status'] == 'completed'
    by_box = {tuple(d['bbox_norm']): d for d in h.items.values()}
    car = next(d for b, d in by_box.items() if b[0] < 0.5)
    truck = next(d for b, d in by_box.items() if b[0] > 0.5)
    assert (car['class_name'], car['class_id']) == ('car', 0)
    assert (truck['class_name'], truck['class_id']) == ('truck', 1)
    for doc in h.items.values():
        assert doc['class_source'] == 'external_label'
        assert doc['class_validated'] is True
        assert doc['label_source'] == 'import'
        assert doc['import_ids'] == [store.import_id]
        assert doc['pe_embedding']  # embedded by the shared index path
        assert doc['embedding_state'] == 'embedded'


@pytest.mark.asyncio
async def test_create_action_materializes_a_real_class_id(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(
        root, names=['car', 'truck'], images={'a': ['0 0.3 0.3 0.2 0.2', '1 0.7 0.7 0.2 0.2']}
    )
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car', 'truck')))
    truck = h.registry.get(1)
    assert truck is not None
    assert truck.class_name == 'truck'
    assert _items_of(h, 'truck')[0]['class_id'] == 1
    assert store.read_mapping()['created_classes'] == {'truck': 1}


@pytest.mark.asyncio
async def test_reimport_never_overwrites_a_locked_human_label(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    h.registry.add_class('truck')
    root = tmp_path / 'ds'
    write_yolo(
        root, names=['car', 'truck'], images={'a': ['0 0.3 0.3 0.2 0.2', '1 0.7 0.7 0.2 0.2']}
    )
    mapping = map_all(h.registry, 'car', 'truck')
    await h.run(h.request(root, mapping))
    car = _items_of(h, 'car')[0]
    cid = car['crop_id']
    # A human relabels the car as a bus.
    h.items[cid].update(
        {'class_id': 99, 'class_name': 'bus', 'class_source': 'human', 'label_source': 'human'}
    )
    # A different dataset version over the same image (a new import key).
    root2 = tmp_path / 'ds2'
    write_yolo(
        root2,
        names=['car', 'truck'],
        images={'a': ['0 0.3 0.3 0.2 0.2', '1 0.7 0.7 0.2 0.2'], 'b': []},
    )
    (root2 / 'images/train/a.jpg').write_bytes((root / 'images/train/a.jpg').read_bytes())
    # same bytes: the image dedups onto the first import's image doc
    store2, _ = await h.run(h.request(root2, mapping, name='v2'))
    after = h.items[cid]
    assert (after['class_id'], after['class_name'], after['class_source']) == (99, 'bus', 'human')
    assert store2.job.read()['report']['label_conflicts_locked'] == 1


@pytest.mark.asyncio
async def test_first_validated_import_writes_region_boxes(tmp_path: Path, monkeypatch) -> None:
    """R2-M1: the class the import just wrote to the parent must not read as a
    pre-existing lock that drops the region boxes."""
    activate_region_profile(parent_classes=('car',))
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(
        root,
        names=['car', 'wheel'],
        images={'a': ['0 0.3 0.3 0.2 0.2', '1 0.3 0.3 0.05 0.05']},
    )
    mapping = [
        *map_all(h.registry, 'car'),
        ClassMappingEntry(dataset_class='wheel', action='region'),
    ]
    store, _ = await h.run(h.request(root, mapping))
    report = store.job.read()['report']
    assert report['boxes_written'] == 1
    assert report['label_conflicts_locked'] == 0
    parent = next(iter(h.items.values()))
    F = get_region_fields()
    assert parent[F.status] == 'detected'
    assert parent[F.validated] is True
    boxes = read_boxes(parent, F)
    assert [(b.source, b.state, b.detector_version) for b in boxes] == [
        ('import', 'accepted', store.import_id)
    ]
    assert parent[F.profile] == 'wheel_profile'


@pytest.mark.asyncio
async def test_region_box_outside_every_parent_becomes_a_standalone_item(
    tmp_path: Path, monkeypatch
) -> None:
    activate_region_profile(parent_classes=('car',))
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(
        root,
        names=['car', 'wheel'],
        images={'a': ['0 0.3 0.3 0.2 0.2', '1 0.8 0.8 0.05 0.05']},
    )
    mapping = [
        *map_all(h.registry, 'car'),
        ClassMappingEntry(dataset_class='wheel', action='region'),
    ]
    store, _ = await h.run(h.request(root, mapping))
    standalone = [d for d in h.items.values() if d.get('import_standalone_region')]
    assert len(standalone) == 1
    s = standalone[0]
    assert s.get('class_id') is None
    assert s['class_validated'] is False
    F = get_region_fields()
    assert [b.state for b in read_boxes(s, F)] == ['accepted']
    assert store.job.read()['report']['standalone_regions'] == 1


@pytest.mark.asyncio
async def test_negatives_unlabeled_and_splits(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(
        root,
        names=['car'],
        images={'pos': ['0 0.5 0.5 0.2 0.2'], 'neg': [], 'nofile': None},
        split='val',
    )
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))
    report = store.job.read()['report']
    assert (report['negatives'], report['unlabeled'], report['items_created']) == (1, 1, 1)
    states = {d['import_source_stem']: d['import_label_state'] for d in h.images.values()}
    assert states == {'pos': 'labeled', 'neg': 'negative', 'nofile': 'unlabeled'}
    neg = next(d for d in h.images.values() if d['import_source_stem'] == 'neg')
    assert neg['negative_for'] == ['car']
    assert all(d['dataset_split'] == 'val' for d in h.images.values())
    assert all(d['dataset_split'] == 'val' for d in h.items.values())
    assert not [d for d in h.items.values() if d['image_id'] == neg['image_id']]


@pytest.mark.asyncio
async def test_freeze_test_split_marks_holdout_and_leaves_current_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    from src.config import get_curation_config

    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'t': ['0 0.5 0.5 0.2 0.2']}, split='test')
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car'), freeze_test_split=True))
    assert all(d['test_holdout'] is True for d in h.items.values())
    holdout_dir = Path(get_curation_config().project_state_dir) / 'test_holdout'
    assert not (holdout_dir / 'current.json').exists()
    assert list((holdout_dir / 'imports').glob('*.json'))
    assert store.job.read()['report']['holdout_frozen'] == 1

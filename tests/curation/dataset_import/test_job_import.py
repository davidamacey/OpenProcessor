"""W10.6/W10.17 (reduced scope): the import job writes real item/image
docs through the same primitives human/ingest writers use, mapping
classes by name."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'integration'))
from ingest_fakes import FakeOpenSearch

from src.clients.curation_opensearch import ClassRegistry
from src.services.curation.dataset_import.job import (
    _image_id_for,
    import_dataset,
    materialize_created_classes,
    registry_class_views,
)
from src.services.curation.dataset_import.mapping import ClassMappingEntry, resolve_mapping
from src.services.curation.dataset_import.yolo import scan_yolo
from src.services.detection.geometry import crop_id as _crop_id


def _write_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (100, 100), color='red').save(path, format='JPEG')


def _make_dataset(root: Path, *, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    names_yaml = '\n'.join(f'  {i}: {n}' for i, n in enumerate(names))
    (root / 'data.yaml').write_text(f'train: images/train\nnames:\n{names_yaml}\n')
    _write_image(root / 'images/train/a.jpg')
    (root / 'labels/train').mkdir(parents=True, exist_ok=True)
    cls_car = names.index('car')
    cls_truck = names.index('truck')
    (root / 'labels/train/a.txt').write_text(
        f'{cls_car} 0.3 0.3 0.2 0.2\n{cls_truck} 0.7 0.7 0.2 0.2\n'
    )


@pytest.mark.asyncio
async def test_import_maps_classes_by_name_not_index(tmp_path: Path) -> None:
    """Dataset ``{0: truck, 1: car}`` against registry ``[car(0), truck(1)]``
    must import trucks as truck -- the pre-W10 bug this wave fixes."""
    dataset_root = tmp_path / 'ds'
    _make_dataset(dataset_root, names=['truck', 'car'])  # index-reversed vs. registry

    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    registry.add_class('car')
    registry.add_class('truck')

    scan = scan_yolo(dataset_root)
    entries = [e for e in ('car', 'truck') if scan.class_box_counts.get(e)]
    mapping_entries = [
        ClassMappingEntry(dataset_class='car', action='map', class_id=0),
        ClassMappingEntry(dataset_class='truck', action='map', class_id=1),
    ]
    resolved = resolve_mapping(
        entries, mapping_entries, registry_classes=registry_class_views(registry)
    )
    assert resolved.ok

    fake_os = FakeOpenSearch()
    report = await import_dataset(
        fake_os,
        scan,
        resolved,
        import_id='imp_test',
        images_index='op_curation_images',
        items_index='op_curation_items',
    )
    assert report.items_created == 2
    classes_written = {doc['class_name'] for doc in fake_os.items.values()}
    assert classes_written == {'car', 'truck'}
    for doc in fake_os.items.values():
        if doc['class_name'] == 'car':
            assert doc['class_id'] == 0
        else:
            assert doc['class_id'] == 1
        assert doc['class_source'] == 'external_label'
        assert doc['class_validated'] is True
        assert doc['label_source'] == 'import'
        assert doc['import_ids'] == ['imp_test']


@pytest.mark.asyncio
async def test_create_action_materializes_a_real_class_id(tmp_path: Path) -> None:
    dataset_root = tmp_path / 'ds'
    _make_dataset(dataset_root, names=['car', 'truck'])
    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    registry.add_class('car')
    # 'truck' has no registry entry -> create.

    scan = scan_yolo(dataset_root)
    mapping_entries = [
        ClassMappingEntry(dataset_class='car', action='map', class_id=0),
        ClassMappingEntry(dataset_class='truck', action='create', new_class_name='truck'),
    ]
    resolved = resolve_mapping(
        ['car', 'truck'], mapping_entries, registry_classes=registry_class_views(registry)
    )
    assert resolved.ok
    materialize_created_classes(resolved, registry)
    assert resolved.targets['truck'].class_id == 1
    created = registry.get(1)
    assert created is not None
    assert created.class_name == 'truck'

    fake_os = FakeOpenSearch()
    report = await import_dataset(
        fake_os,
        scan,
        resolved,
        import_id='imp_test2',
        images_index='op_curation_images',
        items_index='op_curation_items',
    )
    assert report.items_created == 2
    truck_docs = [d for d in fake_os.items.values() if d['class_name'] == 'truck']
    assert truck_docs[0]['class_id'] == 1


@pytest.mark.asyncio
async def test_reimport_never_overwrites_a_locked_human_label(tmp_path: Path) -> None:
    """W10 M2 regression: a human-set label on an existing item must
    survive a re-import that would try to relabel it, and the conflict
    must be reported -- not silently overwritten with class_source still
    reading 'human'."""
    dataset_root = tmp_path / 'ds'
    _make_dataset(dataset_root, names=['car', 'truck'])

    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    registry.add_class('car')
    registry.add_class('truck')

    scan = scan_yolo(dataset_root)
    mapping_entries = [
        ClassMappingEntry(dataset_class='car', action='map', class_id=0),
        ClassMappingEntry(dataset_class='truck', action='map', class_id=1),
    ]
    resolved = resolve_mapping(
        ['car', 'truck'], mapping_entries, registry_classes=registry_class_views(registry)
    )
    assert resolved.ok

    # Pre-seed the items index as if a human had already labeled the
    # 'car' box (0.3 0.3 0.2 0.2 -> bbox 0.2,0.2,0.4,0.4) as 'bus'.
    image_id = _image_id_for(dataset_root / 'images/train/a.jpg')
    car_bbox = (0.2, 0.2, 0.4, 0.4)
    car_crop_id = _crop_id(image_id, list(car_bbox))
    fake_os = FakeOpenSearch()
    fake_os.items[car_crop_id] = {
        'crop_id': car_crop_id,
        'image_id': image_id,
        'class_id': 99,
        'class_name': 'bus',
        'class_source': 'human',
        'label_source': 'human',
        'class_validated': True,
        'import_ids': ['imp_prior'],
    }

    report = await import_dataset(
        fake_os,
        scan,
        resolved,
        import_id='imp_reimport',
        images_index='op_curation_images',
        items_index='op_curation_items',
    )

    # The human label survives untouched.
    after = fake_os.items[car_crop_id]
    assert after['class_id'] == 99
    assert after['class_name'] == 'bus'
    assert after['class_source'] == 'human'
    assert after['import_ids'] == ['imp_prior']

    # The conflict is reported, not silent.
    assert report.items_locked_skipped == 1
    assert any(car_crop_id in c for c in report.conflicts)

    # The unlocked 'truck' box still imports normally.
    assert report.items_created == 1
    truck_docs = [d for d in fake_os.items.values() if d['class_name'] == 'truck']
    assert len(truck_docs) == 1

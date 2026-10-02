"""The crop id an import plans for a labeled box is the id the shared index
path stores it under, even when the label's coordinates sit on a 6-decimal
rounding edge (the index path derives the id from the pixel box)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.harness import Harness, activate_region_profile, map_all, write_yolo
from src.services.curation.dataset_import.mapping import ClassMappingEntry


if TYPE_CHECKING:
    from pathlib import Path


# x1 = 0.517545 - 0.169271 / 2 is 0.4329095 up to float noise: the id differs
# between the label's own value and its pixel round trip on a 100 px image.
EDGE_CAR = '0 0.517545 0.30847 0.169271 0.183381'
WHEEL_INSIDE = '1 0.517545 0.30847 0.05 0.05'


@pytest.mark.asyncio
async def test_region_labels_reach_a_parent_whose_box_sits_on_a_rounding_edge(
    tmp_path: Path, monkeypatch
) -> None:
    activate_region_profile(parent_classes=('car',))
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car', 'wheel'], images={'a': [EDGE_CAR, WHEEL_INSIDE]})
    mapping = [
        *map_all(h.registry, 'car'),
        ClassMappingEntry(dataset_class='wheel', action='region'),
    ]
    store, _ = await h.run(h.request(root, mapping))
    job = store.job.read()
    assert job['status'] == 'completed', job.get('error')
    assert job['report']['boxes_written'] == 1


@pytest.mark.asyncio
async def test_ledger_ids_are_the_stored_item_ids(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': [EDGE_CAR]})
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))
    stored = set(h.items)
    recorded = {i['crop_id'] for row in store.chunk_rows(0).values() for i in row.get('items', [])}
    assert recorded
    assert recorded <= stored


@pytest.mark.asyncio
async def test_undo_removes_the_standalone_region_items_an_import_created(
    tmp_path: Path, monkeypatch
) -> None:
    from src.services.curation.dataset_import.undo import undo_import

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
    assert store.job.read()['report']['standalone_regions'] == 1
    report = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert report.items_deleted == 2
    assert not h.items

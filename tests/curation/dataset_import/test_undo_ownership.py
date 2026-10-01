"""What undo leaves alone: a human verdict on an import's own box, and a
holdout flag the import did not set."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.dataset_import.harness import (
    Harness,
    activate_region_profile,
    map_all,
    write_yolo,
    write_yolo_splits,
)
from src.config.region_fields import get_region_fields
from src.config.region_rejection import REJECT_REASON_HUMAN
from src.services.curation.dataset_import.mapping import ClassMappingEntry
from src.services.curation.dataset_import.undo import undo_import
from src.services.curation.region_boxes import read_boxes


if TYPE_CHECKING:
    from pathlib import Path

SAME = '0 0.3 0.3 0.4 0.4'
F = get_region_fields()
HUMAN = {'class_source': 'human', 'label_source': 'human', 'class_validated': True}


@pytest.mark.asyncio
async def test_a_human_verdict_on_an_untouched_import_box_keeps_the_box(
    tmp_path: Path, monkeypatch: Any
) -> None:
    activate_region_profile(parent_classes=('car',))
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car', 'wheel'], images={'a': [SAME, '1 0.3 0.3 0.05 0.05']})
    mapping = [
        *map_all(h.registry, 'car'),
        ClassMappingEntry(dataset_class='wheel', action='region'),
    ]
    store, _ = await h.run(h.request(root, mapping))
    (cid,) = h.items
    (imported,) = h.items[cid][F.boxes]
    # A reject by a human changes only the verdict stamp: source, detector
    # version and geometry still say "this import wrote it".
    imported.update({'state': 'rejected', 'rejection_reason': REJECT_REASON_HUMAN})

    report = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert (report.boxes_removed, report.boxes_kept_human_edited) == (0, 1)
    assert [b.box_id for b in read_boxes(h.items[cid], F)] == [imported['box_id']]


@pytest.mark.asyncio
async def test_undo_clears_the_holdout_it_set_and_keeps_one_a_curator_set(
    tmp_path: Path, monkeypatch: Any
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo_splits(
        root,
        names=['car'],
        splits={'train': {'tr': [SAME]}, 'test': {'te': [SAME]}},
    )
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car'), freeze_test_split=True))
    by_split = {d['dataset_split']: cid for cid, d in h.items.items()}
    assert h.items[by_split['test']]['test_holdout'] is True
    assert h.items[by_split['train']]['test_holdout'] is False
    # After the import: a human edits both, and a curated freeze takes the train one.
    for cid in by_split.values():
        h.items[cid].update(HUMAN)
    h.items[by_split['train']]['test_holdout'] = True

    report = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert report.holdout_flags_cleared == 1
    assert h.items[by_split['test']]['test_holdout'] is False
    assert h.items[by_split['train']]['test_holdout'] is True


@pytest.mark.asyncio
async def test_undo_of_a_relabel_keeps_a_holdout_a_curator_set_afterwards(
    tmp_path: Path, monkeypatch: Any
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': [SAME]})
    image = root / 'images/train/a.jpg'
    result = await h.service.ingest_one(image.read_bytes(), str(image.resolve()))
    assert result.status == 'success'
    (cid,) = h.items
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))
    assert h.items[cid]['test_holdout'] is False
    h.items[cid]['test_holdout'] = True

    await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert h.items[cid]['test_holdout'] is True

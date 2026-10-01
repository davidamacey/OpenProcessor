"""Import over existing machine items, ``processing: propose``, suggestion
trust and region undo."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.harness import Harness, activate_region_profile, map_all, write_yolo
from src.config.region_fields import get_region_fields
from src.services.curation.dataset_import.mapping import ClassMappingEntry
from src.services.curation.dataset_import.undo import undo_import
from src.services.curation.region_boxes import read_boxes


if TYPE_CHECKING:
    from pathlib import Path

# The fake detector's one box, and a label equal to it (cx, cy, w, h).
DET = (0.1, 0.1, 0.5, 0.5)
SAME = '0 0.3 0.3 0.4 0.4'
FAR = '0 0.8 0.8 0.1 0.1'


async def _ingest_machine_item(h: Harness, image: Path) -> str:
    result = await h.service.ingest_one(image.read_bytes(), str(image.resolve()))
    assert result.status == 'success', result
    (cid,) = h.items
    return cid


@pytest.mark.asyncio
async def test_import_relabels_a_machine_item_with_a_restorable_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': [SAME]})
    cid = await _ingest_machine_item(h, root / 'images/train/a.jpg')
    machine = dict(h.items[cid])
    assert machine['class_validated'] is False

    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))
    assert list(h.items) == [cid]  # same item, relabeled in place
    item = h.items[cid]
    assert (item['class_name'], item['class_validated']) == ('car', True)
    history = item['class_id_history']
    assert [e['writer'] for e in history] == [f'import:{store.import_id}']
    assert history[0]['class_validated'] is False
    ledger = [i for r in store.ledger_rows() for i in r['items']]
    assert [(i['action'], i['snapshot_index']) for i in ledger] == [('updated', 0)]

    report = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert (report.items_restored, report.items_deleted) == (1, 0)
    restored = h.items[cid]
    for field in ('class_source', 'class_validated', 'label_source'):
        assert restored.get(field) == machine.get(field), field
    assert restored.get('class_id') == machine.get('class_id')
    assert restored['import_ids'] == []


@pytest.mark.asyncio
async def test_undo_keeps_a_relabel_a_human_made_after_the_import(
    tmp_path: Path, monkeypatch
) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': [SAME]})
    cid = await _ingest_machine_item(h, root / 'images/train/a.jpg')
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))
    h.items[cid].update(
        {'class_id': 5, 'class_name': 'bus', 'class_source': 'human', 'label_source': 'human'}
    )
    report = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert report.items_restored == 0
    assert (h.items[cid]['class_name'], h.items[cid]['class_source']) == ('bus', 'human')


@pytest.mark.asyncio
async def test_propose_merges_matches_and_creates_the_rest(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'match': [SAME], 'far': [FAR], 'neg': []})
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car'), processing='propose'))
    report = store.job.read()['report']
    imported = [d for d in h.items.values() if d['class_source'] == 'external_label']
    proposals = [d for d in h.items.values() if d.get('proposed_by_import')]
    assert len(imported) == 2
    assert all(d['class_validated'] for d in imported)
    # 'match': the detector box equals the label: noted, not duplicated.
    matched = next(d for d in imported if d['import_source_stem'] == 'match')
    assert matched['proposal_chain'] == ['primary:match']
    # 'far' (label elsewhere) and 'neg' (no label): each gets one machine item.
    assert len(proposals) == 2
    assert all(p['class_validated'] is False for p in proposals)
    on_neg = [p for p in proposals if p.get('on_negative_frame')]
    assert len(on_neg) == 1
    assert report['proposals_created'] == 2
    assert report['proposals_merged'] == 1
    counts = report['disagreement_counts']
    assert counts['missed_labels'] == 1  # the 'far' label no detection overlapped
    # Imported labels are exactly what the dataset said.
    for d in imported:
        assert d['class_name'] == 'car'


@pytest.mark.asyncio
async def test_processing_none_never_calls_the_detector(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    h.registry.add_class('car')
    root = tmp_path / 'ds'
    write_yolo(root, names=['car'], images={'a': [SAME], 'neg': []})
    await h.run(h.request(root, map_all(h.registry, 'car')))
    assert h.triton.batch_sizes == []


@pytest.mark.asyncio
async def test_suggestion_trust_is_not_locked_and_boxes_are_proposed(
    tmp_path: Path, monkeypatch
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
    await h.run(h.request(root, mapping, label_trust='suggestion'))
    (item,) = h.items.values()
    F = get_region_fields()
    assert item['class_validated'] is False
    assert item[F.validated] is False
    assert [b.state for b in read_boxes(item, F)] == ['proposed']
    assert item[F.status] == 'pending_verification'


@pytest.mark.asyncio
async def test_region_undo_removes_import_boxes_but_keeps_a_human_box(
    tmp_path: Path, monkeypatch
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
    F = get_region_fields()
    doc = h.items[cid]
    human = {
        'box_id': 'b2',
        'bbox_norm': [0.2, 0.2, 0.25, 0.25],
        'state': 'accepted',
        'source': 'human',
        'detector': 'human',
    }
    doc[F.boxes] = [*doc[F.boxes], human]

    report = await undo_import(h.undo_context(store.import_id), store, dry_run=False)
    assert report.boxes_removed == 1
    assert report.items_kept_human_edited == 1
    boxes = read_boxes(h.items[cid], F)
    assert [(b.box_id, b.source) for b in boxes] == [('b2', 'human')]
    assert h.items[cid][F.status] == 'detected'

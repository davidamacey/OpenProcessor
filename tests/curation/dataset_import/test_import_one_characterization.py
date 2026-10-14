"""Pins the per-image import (ledger rows, report counters and the stored
documents) across the paths ``_import_one`` takes: labeled, negative,
unlabeled and all-skipped frames, unreadable and size-mismatched images,
regions with label and detected parents, standalone regions, proposals,
test-split freezing, re-imports over locked/changed/unchanged labels and a
resumed image whose write-ahead row already exists.

The golden file was captured from the implementation before ``_import_one``
was decomposed.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from curation.dataset_import.harness import (
    Harness,
    activate_region_profile,
    map_all,
    write_yolo,
    write_yolo_splits,
)
from src.services.curation.dataset_import import runner
from src.services.curation.dataset_import.chunk import import_chunk
from src.services.curation.dataset_import.mapping import ClassMappingEntry


GOLDEN = Path(__file__).parent / 'golden' / 'import_one_golden.json'
ITEM_KEYS = (
    'image_id',
    'class_name',
    'class_id',
    'class_source',
    'class_validated',
    'label_source',
    'import_ids',
    'import_source_stem',
    'import_label_state',
    'dataset_split',
    'proposed_by_import',
    'on_negative_frame',
    'test_holdout',
    'bbox_norm',
    'region_boxes',
    'proposal_chain',
    'embedding_state',
)
IMAGE_KEYS = (
    'import_ids',
    'import_source_stem',
    'import_label_state',
    'dataset_split',
    'negative_for',
    'import_stratum',
)
SAME = '0 0.3 0.3 0.4 0.4'
FAR = '0 0.8 0.8 0.1 0.1'


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _clean(v) for k, v in value.items() if not (k.endswith('_at') or k in ('ts', 'time'))
        }
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _stable_ids(h: Harness) -> dict[str, str]:
    """Image and crop ids are random per run; name them by stem and box instead."""
    names: dict[str, str] = {}
    for image_id, doc in h.images.items():
        names[image_id] = f'IMG[{doc.get("import_source_stem")}]'
    for crop_id, doc in h.items.items():
        box = ','.join(f'{v:.3f}' for v in doc.get('bbox_norm') or [])
        names[crop_id] = f'CROP[{names.get(doc.get("image_id"), doc.get("image_id"))}|{box}]'
    return names


def snapshot(h: Harness, store: Any, tmp: Path) -> dict[str, Any]:
    snap = {
        'report': store.job.read().get('report'),
        'ledger': list(store.ledger_rows()),
        'items': {cid: {k: d[k] for k in ITEM_KEYS if k in d} for cid, d in h.items.items()},
        'images': {iid: {k: d[k] for k in IMAGE_KEYS if k in d} for iid, d in h.images.items()},
    }
    text = json.dumps(_clean(snap), sort_keys=True, default=str)
    for raw, name in sorted(_stable_ids(h).items(), key=lambda kv: -len(kv[0])):
        text = text.replace(raw, name)
    text = text.replace(str(tmp), '<tmp>')
    text = re.sub(r'imp_\d{8}T\d{6}_[0-9a-f]{8}', '<import>', text)
    return json.loads(text)


async def _basic(h: Harness, tmp: Path) -> dict[str, Any]:
    h.registry.add_class('car')
    root = tmp / 'ds'
    write_yolo(
        root,
        names=['car', 'truck', 'junk'],
        images={
            'both': ['0 0.3 0.3 0.2 0.2', '1 0.7 0.7 0.2 0.2'],
            'neg': [],
            'unl': None,
            'skipped': ['2 0.5 0.5 0.2 0.2'],
        },
    )
    mapping = [
        *map_all(h.registry, 'car', 'truck'),
        ClassMappingEntry(dataset_class='junk', action='skip'),
    ]
    store, _ = await h.run(h.request(root, mapping))
    return snapshot(h, store, tmp)


async def _unreadable(h: Harness, tmp: Path) -> dict[str, Any]:
    h.registry.add_class('car')
    root = tmp / 'ds'
    write_yolo(
        root, names=['car'], images={'ok': ['0 0.3 0.3 0.2 0.2'], 'bad': ['0 0.5 0.5 0.2 0.2']}
    )
    (root / 'images/train/bad.jpg').write_bytes(b'not an image')
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car')))
    return snapshot(h, store, tmp)


async def _size_mismatch(h: Harness, tmp: Path) -> dict[str, Any]:
    h.registry.add_class('car')
    root = tmp / 'ds'
    write_yolo(root, names=['car'], images={'a': ['0 0.3 0.3 0.2 0.2'], 'b': ['0 0.5 0.5 0.2 0.2']})
    request = h.request(root, map_all(h.registry, 'car'))
    prepared = h.prepare(request)
    store, _ = runner.claim_import(request, prepared)
    h.last_store = store
    runner.persist_pinned(store, prepared)
    ctx = h.context(store, request, prepared)
    entries = runner.ordered_entries(prepared.scan.entries)
    entries[0] = dataclasses.replace(entries[0], declared_size=(7, 7))
    entries[1] = dataclasses.replace(entries[1], declared_size=(100, 100))
    report = await import_chunk(ctx, store, 0, entries)
    snap = snapshot(h, store, tmp)
    snap['chunk_report'] = report.to_dict()
    return snap


async def _regions_label_parents(h: Harness, tmp: Path) -> dict[str, Any]:
    activate_region_profile(parent_classes=('car',))
    h.registry.add_class('car')
    root = tmp / 'ds'
    write_yolo(
        root,
        names=['car', 'wheel'],
        images={
            'inside': ['0 0.3 0.3 0.4 0.4', '1 0.3 0.3 0.05 0.05'],
            'outside': ['0 0.3 0.3 0.2 0.2', '1 0.9 0.9 0.05 0.05'],
            'bare': ['0 0.6 0.6 0.2 0.2'],
        },
    )
    mapping = [
        *map_all(h.registry, 'car'),
        ClassMappingEntry(dataset_class='wheel', action='region'),
    ]
    store, _ = await h.run(h.request(root, mapping, parents='labels', region_negatives=True))
    return snapshot(h, store, tmp)


async def _regions_detected_parents(h: Harness, tmp: Path) -> dict[str, Any]:
    activate_region_profile()
    h.registry.add_class('car')
    root = tmp / 'ds'
    write_yolo(
        root,
        names=['car', 'wheel'],
        images={'a': ['1 0.3 0.3 0.05 0.05'], 'b': ['1 0.9 0.9 0.05 0.05'], 'neg': []},
    )
    mapping = [ClassMappingEntry(dataset_class='wheel', action='region')]
    store, _ = await h.run(h.request(root, mapping, parents='detect'))
    return snapshot(h, store, tmp)


async def _propose(h: Harness, tmp: Path) -> dict[str, Any]:
    h.registry.add_class('car')
    root = tmp / 'ds'
    write_yolo(root, names=['car'], images={'match': [SAME], 'far': [FAR], 'neg': [], 'unl': None})
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car'), processing='propose'))
    return snapshot(h, store, tmp)


async def _freeze_test_split(h: Harness, tmp: Path) -> dict[str, Any]:
    h.registry.add_class('car')
    root = tmp / 'ds'
    write_yolo_splits(
        root,
        names=['car'],
        splits={'train': {'t1': [SAME]}, 'test': {'e1': [SAME, FAR], 'e2': []}},
    )
    store, _ = await h.run(h.request(root, map_all(h.registry, 'car'), freeze_test_split=True))
    return snapshot(h, store, tmp)


async def _freeze_updated_item(h: Harness, tmp: Path) -> dict[str, Any]:
    h.registry.add_class('car')
    h.registry.add_class('truck')
    first = tmp / 'ds1'
    write_yolo_splits(first, names=['car', 'truck'], splits={'test': {'e1': [SAME]}})
    await h.run(h.request(first, map_all(h.registry, 'car'), freeze_test_split=False))
    second = tmp / 'ds2'
    write_yolo_splits(
        second, names=['car', 'truck'], splits={'test': {'e1': ['1 0.3 0.3 0.4 0.4']}}
    )
    store, _ = await h.run(
        h.request(second, map_all(h.registry, 'truck'), freeze_test_split=True, name='v2')
    )
    return snapshot(h, store, tmp)


async def _propose_with_detected_parents(h: Harness, tmp: Path) -> dict[str, Any]:
    activate_region_profile()
    root = tmp / 'ds'
    write_yolo(root, names=['wheel'], images={'a': ['0 0.3 0.3 0.05 0.05'], 'neg': []})
    mapping = [ClassMappingEntry(dataset_class='wheel', action='region')]
    store, _ = await h.run(h.request(root, mapping, parents='detect', processing='propose'))
    return snapshot(h, store, tmp)


async def _reimport(h: Harness, tmp: Path) -> dict[str, Any]:
    h.registry.add_class('car')
    h.registry.add_class('truck')
    root = tmp / 'ds'
    rows = ['0 0.2 0.2 0.1 0.1', '1 0.5 0.5 0.1 0.1', '0 0.8 0.8 0.1 0.1']
    write_yolo(root, names=['car', 'truck'], images={'a': rows})
    mapping = map_all(h.registry, 'car', 'truck')
    await h.run(h.request(root, mapping))
    first = min(h.items, key=lambda c: h.items[c]['bbox_norm'][0])
    h.items[first].update({'class_id': 99, 'class_name': 'bus', 'class_source': 'human'})
    root2 = tmp / 'ds2'
    write_yolo(
        root2,
        names=['car', 'truck'],
        images={'a': [rows[0], '0 0.5 0.5 0.1 0.1', rows[2]]},
    )
    (root2 / 'images/train/a.jpg').write_bytes((root / 'images/train/a.jpg').read_bytes())
    store, _ = await h.run(h.request(root2, map_all(h.registry, 'car'), name='v2'))
    return snapshot(h, store, tmp)


async def _resumed_image(h: Harness, tmp: Path) -> dict[str, Any]:
    h.registry.add_class('car')
    root = tmp / 'ds'
    write_yolo(root, names=['car'], images={'a': [SAME, FAR], 'b': ['0 0.5 0.5 0.1 0.1']})
    request = h.request(root, map_all(h.registry, 'car'))
    prepared = h.prepare(request)
    store, _ = runner.claim_import(request, prepared)
    h.last_store = store
    runner.persist_pinned(store, prepared)
    ctx = h.context(store, request, prepared)
    entries = runner.ordered_entries(prepared.scan.entries)
    await import_chunk(ctx, store, 0, entries)
    # The process died after the write-ahead row and the item writes: keep only
    # the pending rows (their recorded actions say 'created') and re-run.
    ledger = store.directory / 'ledger' / '000000.jsonl'
    pending = [
        r for r in map(json.loads, ledger.read_text().splitlines()) if r['status'] == 'pending'
    ]
    ledger.write_text(''.join(json.dumps(r) + '\n' for r in pending))
    report = await import_chunk(ctx, store, 0, entries)
    snap = snapshot(h, store, tmp)
    snap['rerun_report'] = report.to_dict()
    return snap


SCENARIOS = {
    'basic': _basic,
    'unreadable': _unreadable,
    'size_mismatch': _size_mismatch,
    'regions_label_parents': _regions_label_parents,
    'regions_detected_parents': _regions_detected_parents,
    'propose': _propose,
    'freeze_test_split': _freeze_test_split,
    'freeze_updated_item': _freeze_updated_item,
    'propose_with_detected_parents': _propose_with_detected_parents,
    'reimport': _reimport,
    'resumed_image': _resumed_image,
}


@pytest.mark.asyncio
@pytest.mark.parametrize('name', sorted(SCENARIOS))
async def test_import_matches_golden(name: str, tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path, monkeypatch)
    got = await SCENARIOS[name](h, tmp_path)
    assert got == json.loads(GOLDEN.read_text())[name]


def test_golden_file_has_exactly_the_scenarios() -> None:
    assert set(json.loads(GOLDEN.read_text())) == set(SCENARIOS)


def test_golden_reaches_every_ledger_status_and_action() -> None:
    golden = json.loads(GOLDEN.read_text())
    statuses = {r['status'] for g in golden.values() for r in g['ledger']}
    actions = {
        i['action'] for g in golden.values() for r in g['ledger'] for i in r.get('items', [])
    }
    kinds = {r.get('error_kind') for g in golden.values() for r in g['ledger']}
    assert {'ok', 'failed', 'skipped'} <= statuses
    assert {'created', 'noop', 'updated', 'standalone', 'parent', 'proposal'} <= actions
    assert {'image_size_mismatch'} <= kinds

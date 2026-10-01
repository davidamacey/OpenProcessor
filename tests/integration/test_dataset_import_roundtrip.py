"""An OpenProcessor export re-imports faithfully (W10.2.2): into a fresh
project the labels, splits and frozen test set round-trip; into the project
that exported it nothing is written."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from curation.dataset_import.harness import Harness, write_yolo_splits

from src.config.curation import CurationConfig
from src.services.curation.dataset_import.mapping import ClassMappingEntry
from src.services.curation.export import GenericYoloExportService


SPLITS = {
    'train': {
        't1': ['0 0.3 0.3 0.2 0.2', '1 0.7 0.7 0.2 0.2'],
        't2': ['0 0.5 0.5 0.3 0.3'],
        't3': ['1 0.4 0.4 0.2 0.2'],
    },
    'val': {'v1': ['0 0.2 0.2 0.1 0.1']},
    'test': {'x1': ['1 0.6 0.6 0.2 0.2'], 'x2': ['0 0.5 0.5 0.4 0.4']},
}


def _mapping(h: Harness, root: Path) -> list[ClassMappingEntry]:
    out = []
    for name, s in h.prepare(h.request(root)).suggestions.items():
        if s.action == 'map':
            out.append(ClassMappingEntry(dataset_class=name, action='map', class_id=s.class_id))
        else:
            out.append(ClassMappingEntry(dataset_class=name, action='create', new_class_name=name))
    return out


async def _export(h: Harness, tag: str, tmp: Path) -> Path:
    cfg = CurationConfig(
        items_index=h.cfg.items_index,
        images_index=h.cfg.images_index,
        export_root=tmp / f'exports_{tag}',
    )
    service = GenericYoloExportService(h.os, config=cfg, registry=h.registry)
    result = await service.export_dataset(version_tag=tag, copy_images=True)
    return h_export_dir(result)


def h_export_dir(result) -> Path:
    return Path(result.export_dir)


def _rows_by_stem(export: Path, stem_of_label: dict[str, str]) -> dict[str, tuple[str, Counter]]:
    """``{source stem: (split, multiset of 'name cx cy w h' rows)}``."""
    data = (export / 'data.yaml').read_text()
    names_line = next(line for line in data.splitlines() if line.startswith('names:'))
    names = json.loads(names_line.removeprefix('names:').strip())
    out: dict[str, tuple[str, Counter]] = {}
    for label in sorted(export.glob('labels/*/*.txt')):
        rows: Counter[str] = Counter()
        for line in label.read_text().splitlines():
            if line.strip():
                cls, *rest = line.split()
                rows[' '.join([names[int(cls)], *rest])] += 1
        out[stem_of_label.get(label.stem, label.stem)] = (label.parent.name, rows)
    return out


@pytest.mark.asyncio
async def test_export_reimports_into_a_fresh_project_and_round_trips(
    tmp_path: Path, monkeypatch
) -> None:
    a = Harness(tmp_path / 'a', monkeypatch, root=tmp_path)
    src = tmp_path / 'src'
    write_yolo_splits(src, names=['car', 'truck'], splits=SPLITS)
    store_a, _ = await a.run(a.request(src, _mapping(a, src), freeze_test_split=True))
    assert store_a.job.read()['status'] == 'completed'
    export_a = await _export(a, 'a', tmp_path)
    manifest_a = json.loads((export_a / 'manifest.json').read_text())
    # The imported test split is kept, not recomputed by hash.
    split_of = {d['import_source_stem']: d['dataset_split'] for d in a.images.values()}
    exported_a = _rows_by_stem(export_a, {})  # keyed by the exporter's stem: A's image id
    by_original = {d['import_source_stem']: exported_a[d['image_id']] for d in a.images.values()}
    assert {stem: split for stem, (split, _r) in by_original.items()} == split_of

    b = Harness(tmp_path / 'b', monkeypatch, root=tmp_path)
    store_b, _ = await b.run(b.request(export_a, _mapping(b, export_a)))
    state_b = store_b.job.read()
    assert state_b['status'] == 'completed', state_b
    assert state_b['source_format'] == 'openprocessor_export'
    # The export's frozen test split is kept by default (W10.2.2).
    test_items = [d for d in b.items.values() if d['dataset_split'] == 'test']
    assert test_items
    assert all(d['test_holdout'] is True for d in test_items)
    export_b = await _export(b, 'b', tmp_path)
    exported_b = _rows_by_stem(
        export_b, {d['image_id']: d['import_source_stem'] for d in b.images.values()}
    )
    assert exported_b == exported_a
    manifest_b = json.loads((export_b / 'manifest.json').read_text())
    assert manifest_b.get('source_frozen_test_sha') == manifest_a.get('frozen_test_sha')


@pytest.mark.asyncio
async def test_export_reimports_into_its_own_project_writing_no_labels(
    tmp_path: Path, monkeypatch
) -> None:
    a = Harness(tmp_path / 'a', monkeypatch, root=tmp_path)
    src = tmp_path / 'src'
    write_yolo_splits(src, names=['car', 'truck'], splits=SPLITS)
    await a.run(a.request(src, _mapping(a, src)))
    export_a = await _export(a, 'a', tmp_path)
    before = {k: dict(v) for k, v in a.items.items()}
    n_images = len(a.images)
    store, _ = await a.run(a.request(export_a, _mapping(a, export_a)))
    report = store.job.read()['report']
    assert report['items_created'] == 0
    assert report['items_updated'] == 0
    assert report['images_created'] == 0
    assert len(a.images) == n_images  # the exported copies were not ingested
    for cid, doc in before.items():
        for field in ('class_id', 'class_name', 'class_source', 'bbox_norm', 'class_validated'):
            assert a.items[cid][field] == doc[field], field

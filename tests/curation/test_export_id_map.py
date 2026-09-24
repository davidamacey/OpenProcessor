"""Coverage for the dense ``export_id_map`` (plan §4 Wave 3, W3.a).

Registry id -> contiguous-from-zero dense id, deprecated classes skipped;
the label files on disk must carry the dense id, not the raw registry id;
and the manifest's recorded map must be invertible (dense -> original
registry id recoverable).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig
from src.services.curation.export import (
    GenericYoloExportService,
    _build_export_id_map,
    _ExportRow,
    _remap_rows_to_export_ids,
)


class _FakeOpenSearch:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        hits = [{'_id': d['crop_id'], '_source': d} for d in self._docs]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits}}

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None


def test_build_export_id_map_is_contiguous_and_skips_deprecated(tmp_path):
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('car')  # id 0
    reg.add_class('truck')  # id 1
    reg.add_class('bus')  # id 2
    reg.rename_class(1, 'truck')  # no-op, keep id stable

    # Deprecate the middle class by writing it back with deprecated=True.
    loaded = reg.load()
    for c in loaded.classes:
        if c.class_id == 1:
            c.deprecated = True
    reg._atomic_write(loaded)

    id_map = _build_export_id_map(reg.load().classes)

    assert 1 not in id_map  # deprecated class excluded
    assert sorted(id_map.values()) == list(range(len(id_map)))  # contiguous from 0
    assert id_map[0] < id_map[2]  # ascending registry-id order preserved


def test_remap_rows_drops_unmapped_classes_and_sets_export_id():
    rows = [
        _ExportRow(
            item_id='a',
            image_id='img-a',
            image_path='a.jpg',
            bbox_norm=[0, 0, 1, 1],
            class_id=0,
            class_name='car',
        ),
        _ExportRow(
            item_id='b',
            image_id='img-b',
            image_path='b.jpg',
            bbox_norm=[0, 0, 1, 1],
            class_id=99,
            class_name='ghost',
        ),
    ]
    id_map = {0: 0}
    kept = _remap_rows_to_export_ids(rows, id_map)

    assert [r.item_id for r in kept] == ['a']
    assert kept[0].export_class_id == 0


@pytest.mark.asyncio
async def test_label_files_on_disk_use_dense_ids_not_registry_ids(tmp_path):
    reg = ClassRegistry(path=tmp_path / 'registry' / 'class_registry.json')
    reg.add_class('car')  # registry id 0
    reg.add_class('truck')  # registry id 1
    reg.rename_class(0, 'car')

    # Deprecate registry id 0 so registry id 1 ("truck") gets dense id 0,
    # not 1 -- proves the label file reflects the DENSE id, not the raw one.
    loaded = reg.load()
    for c in loaded.classes:
        if c.class_id == 0:
            c.deprecated = True
    reg._atomic_write(loaded)

    docs = [
        {
            'crop_id': 'crop-truck',
            'image_id': 'img-1',
            'image_path': 'truck.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 1,
            'class_name': 'truck',
        }
    ]
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    service = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg, registry=reg)
    result = await service.export_dataset()

    # One label file per source image, named by image_id.
    label_files = list(Path(result.export_dir).glob('labels/*/img-1.txt'))
    assert len(label_files) == 1
    written_id = int(label_files[0].read_text().split()[0])
    assert written_id == 0  # dense id, not registry id 1

    registry_payload = json.loads((Path(result.export_dir) / 'class_registry.json').read_text())
    assert registry_payload['export_id_map']['1'] == 0
    assert '0' not in registry_payload['export_id_map']  # deprecated, excluded


@pytest.mark.asyncio
async def test_manifest_export_id_map_is_invertible(tmp_path):
    reg = ClassRegistry(path=tmp_path / 'registry' / 'class_registry.json')
    reg.add_class('car')
    reg.add_class('truck')
    reg.add_class('bus')

    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_id': f'img-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': i,
            'class_name': name,
        }
        for i, name in enumerate(['car', 'truck', 'bus'])
    ]
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    service = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg, registry=reg)
    result = await service.export_dataset()

    payload = json.loads((Path(result.export_dir) / 'class_registry.json').read_text())
    export_id_map: dict[str, int] = payload['export_id_map']
    inverse = {v: int(k) for k, v in export_id_map.items()}

    # Every dense id maps back to exactly one original registry id.
    assert len(inverse) == len(export_id_map)
    for registry_id_str, dense_id in export_id_map.items():
        assert inverse[dense_id] == int(registry_id_str)


@pytest.mark.asyncio
async def test_manifest_counts_rows_dropped_for_unregistered_class_ids(tmp_path):
    """A validated item carrying a class id the registry doesn't have (e.g.
    a candidate cluster id written as a class) must never become a label,
    and the drop must be visible in the manifest, not only in a log line."""
    reg = ClassRegistry(path=tmp_path / 'registry' / 'class_registry.json')
    reg.add_class('car')
    docs = [
        {
            'crop_id': cid,
            'image_id': f'img-{cid}',
            'image_path': f'{cid}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': class_id,
            'class_name': name,
        }
        for cid, class_id, name in [('ok', 0, 'car'), ('ghost1', 10000, ''), ('ghost2', 10000, '')]
    ]
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    service = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg, registry=reg)
    result = await service.export_dataset()

    manifest = json.loads((Path(result.export_dir) / 'manifest.json').read_text())
    assert manifest['image_count'] == 1
    assert manifest['dropped_unregistered_class_ids'] == {'10000': 2}
    assert not list(Path(result.export_dir).glob('labels/*/ghost*.txt'))

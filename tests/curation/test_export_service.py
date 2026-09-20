"""Unit tests for :mod:`src.services.curation.export`.

Fake OpenSearch (scroll + clear_scroll) and a tmp_path export root — no
real cluster, no real image files (label writing only touches the
filesystem's label/.txt half of the pipeline).
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
    dataset_checksum,
    hash_split,
    resolve_current_export_dir,
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


def _make_registry(base_dir: Path, names: list[str]) -> ClassRegistry:
    reg = ClassRegistry(path=base_dir / 'class_registry.json')
    for name in names:
        reg.add_class(name)
    return reg


def _service(
    tmp_path: Path, docs: list[dict[str, Any]], names: list[str]
) -> GenericYoloExportService:
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    registry = _make_registry(tmp_path / 'registry', names)
    return GenericYoloExportService(_FakeOpenSearch(docs), config=cfg, registry=registry)


def test_hash_split_is_deterministic():
    a = hash_split('crop-1', seed=42, train_ratio=0.8, val_ratio=0.1)
    b = hash_split('crop-1', seed=42, train_ratio=0.8, val_ratio=0.1)
    assert a == b
    assert a in {'train', 'val', 'test'}


def test_hash_split_changes_with_seed_sometimes():
    # Not a strict guarantee for every key, but across many keys at least
    # one seed change should move at least one key to a different bucket.
    keys = [f'crop-{i}' for i in range(200)]
    a = {k: hash_split(k, seed=1, train_ratio=0.8, val_ratio=0.1) for k in keys}
    b = {k: hash_split(k, seed=2, train_ratio=0.8, val_ratio=0.1) for k in keys}
    assert a != b


def test_dataset_checksum_is_order_independent():
    assert dataset_checksum(['a', 'b', 'c']) == dataset_checksum(['c', 'a', 'b'])
    assert dataset_checksum(['a', 'b']) != dataset_checksum(['a', 'b', 'c'])


def test_resolve_current_export_dir_missing_raises(tmp_path):
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    with pytest.raises(FileNotFoundError):
        resolve_current_export_dir(cfg)


@pytest.mark.asyncio
async def test_export_dataset_writes_manifest_and_labels(tmp_path):
    docs = [
        {
            'crop_id': 'crop-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
            'test_holdout': True,
        },
        {
            'crop_id': 'crop-2',
            'image_path': 'b.jpg',
            'bbox_norm': [0.2, 0.2, 0.6, 0.6],
            'class_id': 1,
            'class_name': 'truck',
            'test_holdout': False,
        },
    ]
    service = _service(tmp_path, docs, ['car', 'truck'])

    result = await service.export_dataset(version_tag='t1', seed=7)

    assert result.image_count == 2
    assert result.class_count == 2
    assert result.split_counts.test == 1  # crop-1's test_holdout=True is honored

    current_link = tmp_path / 'exports' / 'current'
    assert current_link.resolve() == Path(result.export_dir).resolve()

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['dataset_sha'] == result.dataset_sha
    assert manifest['image_count'] == 2


@pytest.mark.asyncio
async def test_export_dataset_respects_max_images(tmp_path):
    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': 0,
            'class_name': 'car',
        }
        for i in range(5)
    ]
    service = _service(tmp_path, docs, ['car'])

    result = await service.export_dataset(max_images=2)

    assert result.image_count == 2


# ---------------------------------------------------------------------------
# W3.a restored cases (plan §4 Wave 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_yaml_nc_matches_class_count(tmp_path):
    docs = [
        {
            'crop_id': 'crop-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    service = _service(tmp_path, docs, ['car', 'truck', 'bus'])

    result = await service.export_dataset()

    data_yaml = (Path(result.export_dir) / 'data.yaml').read_text()
    assert 'nc: 3' in data_yaml
    assert result.class_count == 3


@pytest.mark.asyncio
async def test_frozen_holdout_rows_land_in_test_split(tmp_path):
    """``export.py``'s own module docstring advertises this; pin it."""
    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': 0,
            'class_name': 'car',
            # Every row hashes wherever it hashes EXCEPT this one, which is
            # frozen -- it must land in test regardless of the hash bucket.
            'test_holdout': i == 0,
        }
        for i in range(20)
    ]
    service = _service(tmp_path, docs, ['car'])

    result = await service.export_dataset(seed=42)

    label_path = Path(result.export_dir) / 'labels' / 'test' / 'crop-0.txt'
    assert label_path.is_file()


def test_region_bbox_round_trip():
    """normalized bbox -> YOLO cx/cy/w/h text -> back to a normalized bbox."""
    from src.services.curation.export import _write_yolo_label

    bbox = [0.1, 0.2, 0.5, 0.6]

    def _yolo_to_norm(cx, cy, w, h):
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    tmp_txt = Path('/tmp') / 'round_trip_test.txt'
    try:
        _write_yolo_label(tmp_txt, 3, bbox)
        parts = tmp_txt.read_text().split()
        cid = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:])
        recovered = _yolo_to_norm(cx, cy, w, h)
        assert cid == 3
        for a, b in zip(bbox, recovered, strict=True):
            assert a == pytest.approx(b, abs=1e-5)
    finally:
        tmp_txt.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_manifest_structure(tmp_path):
    docs = [
        {
            'crop_id': 'crop-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    service = _service(tmp_path, docs, ['car'])
    result = await service.export_dataset(seed=99)
    manifest = json.loads(Path(result.manifest_path).read_text())
    for key in (
        'version_tag',
        'seed',
        'dataset_sha',
        'image_count',
        'split_counts',
        'class_count',
        'started_at',
        'finished_at',
    ):
        assert key in manifest
    assert manifest['dataset_sha'] == dataset_checksum(['crop-1'])


@pytest.mark.asyncio
async def test_current_symlink_resolves_to_export_dir(tmp_path):
    docs = [
        {
            'crop_id': 'crop-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    service = _service(tmp_path, docs, ['car'])
    result = await service.export_dataset()
    cfg = service.config
    resolved = resolve_current_export_dir(cfg)
    assert resolved == Path(result.export_dir).resolve()


@pytest.mark.asyncio
async def test_export_is_rederivable_from_recorded_seed(tmp_path):
    """Re-running the export with the manifest's recorded seed against the
    same item pool reproduces byte-identical labels (modulo the
    timestamp fields the manifest itself carries)."""
    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': i % 2,
            'class_name': 'car' if i % 2 == 0 else 'truck',
            'test_holdout': i == 0,
        }
        for i in range(30)
    ]

    def _label_bytes(export_dir: Path) -> dict[str, bytes]:
        out = {}
        for split in ('train', 'val', 'test'):
            for f in sorted((export_dir / 'labels' / split).glob('*.txt')):
                out[f'{split}/{f.name}'] = f.read_bytes()
        return out

    cfg1 = CurationConfig(export_root=tmp_path / 'run1')
    registry1 = _make_registry(tmp_path / 'reg1', ['car', 'truck'])
    service1 = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg1, registry=registry1)
    result1 = await service1.export_dataset(seed=123)
    manifest1 = json.loads(Path(result1.manifest_path).read_text())

    cfg2 = CurationConfig(export_root=tmp_path / 'run2')
    registry2 = _make_registry(tmp_path / 'reg2', ['car', 'truck'])
    service2 = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg2, registry=registry2)
    result2 = await service2.export_dataset(seed=manifest1['seed'])

    assert _label_bytes(Path(result1.export_dir)) == _label_bytes(Path(result2.export_dir))
    assert result1.dataset_sha == result2.dataset_sha
    assert result1.split_counts.to_dict() == result2.split_counts.to_dict()

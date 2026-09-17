"""Unit tests for :mod:`src.services.curation.export`.

Fake OpenSearch (scroll + clear_scroll) and a tmp_path export root — no
real cluster, no real image files (label writing only touches the
filesystem's label/.txt half of the pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    service = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg)

    result = await service.export_dataset(version_tag='t1', seed=7)

    assert result.image_count == 2
    assert result.class_count == 2
    assert result.split_counts.test == 1  # crop-1's test_holdout=True is honored

    current_link = tmp_path / 'exports' / 'current'
    assert current_link.resolve() == Path(result.export_dir).resolve()

    import json

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
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    service = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg)

    result = await service.export_dataset(max_images=2)

    assert result.image_count == 2

"""Split assignment for the multi-class export: leakage unit, frozen
holdout, small classes, determinism, and the per-class split counts the
export records and ``GET /export/status`` serves.

The regression this pins: clustering assigns class-sized clusters
(``cluster_id == class_id``), so grouping the split on ``cluster_id``
put each class's validated items into ONE group. A single frozen holdout
row in that group then dragged the whole class into ``test`` — an export
of 174 validated items across 5 classes came out ``train=0, val=0,
test=174``. The split now groups on ``image_id`` (the real leakage unit)
and splits every non-holdout group of a class with a frozen holdout
between train and val only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig
from src.services.curation.export import (
    GenericYoloExportService,
    _ExportRow,
    resolve_current_export_dir,
    stratified_split,
)


CLASS_NAMES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
HOLDOUT_PER_CLASS = 5


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


def _service(tmp_path: Path, docs: list[dict[str, Any]]) -> GenericYoloExportService:
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    registry = ClassRegistry(path=tmp_path / 'registry' / 'class_registry.json')
    for name in CLASS_NAMES:
        registry.add_class(name)
    return GenericYoloExportService(_FakeOpenSearch(docs), config=cfg, registry=registry)


def _doc(
    crop_id: str, image_id: str, class_id: int, *, holdout: bool = False, cluster_id: int = -1
) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_id': image_id,
        'image_path': f'{image_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': class_id,
        'class_name': CLASS_NAMES[class_id],
        'test_holdout': holdout,
        # Class-sized semantic cluster, exactly as clustering assigns it.
        'cluster_id': class_id if cluster_id == -1 else cluster_id,
    }


def _class_sized_cluster_docs() -> list[dict[str, Any]]:
    """Five classes (34 items in the first, 35 in each other), one class-sized
    cluster each, 5 frozen holdout items per class, every item on its own
    image — plus one non-holdout ``gamma`` item sharing a source image with
    a ``beta`` holdout item (a same-image mate)."""
    docs = [
        _doc(f'c{c}-{i:02d}', f'img-c{c}-{i:02d}', c, holdout=i < HOLDOUT_PER_CLASS)
        for c in range(len(CLASS_NAMES))
        for i in range(34 if c == 0 else 35)
    ]
    docs.append(_doc('mate-gamma', 'img-c1-00', 2))
    return docs


def _split_by_item(export_dir: Path, docs: list[dict[str, Any]]) -> dict[str, str]:
    """Item id -> split, read back from the per-image label files on disk.

    Each image is in exactly one split, and holds one label line per item
    on it."""
    image_split: dict[str, str] = {}
    lines_per_image: dict[str, int] = {}
    for split in ('train', 'val', 'test'):
        for f in (export_dir / 'labels' / split).glob('*.txt'):
            assert f.stem not in image_split, f'{f.stem} is in more than one split'
            image_split[f.stem] = split
            lines_per_image[f.stem] = len(f.read_text().splitlines())
    items_per_image: dict[str, int] = defaultdict(int)
    for d in docs:
        items_per_image[d['image_id']] += 1
    assert lines_per_image == dict(items_per_image)
    return {d['crop_id']: image_split[d['image_id']] for d in docs}


# ------------------------------------------------------------- the regression


@pytest.mark.asyncio
async def test_class_sized_clusters_with_a_holdout_still_yield_train_and_val(
    tmp_path: Path,
) -> None:
    docs = _class_sized_cluster_docs()
    result = await _service(tmp_path, docs).export_dataset(seed=42, copy_images=False)

    assert result.split_counts.train > 0
    assert result.split_counts.val > 0

    split_of = _split_by_item(Path(result.export_dir), docs)
    holdout_ids = {d['crop_id'] for d in docs if d['test_holdout']}
    holdout_images = {d['image_id'] for d in docs if d['test_holdout']}
    mates = {
        d['crop_id'] for d in docs if not d['test_holdout'] and d['image_id'] in holdout_images
    }
    assert mates == {'mate-gamma'}

    # test is exactly the frozen holdout plus its same-image mates.
    test_ids = {cid for cid, s in split_of.items() if s == 'test'}
    assert test_ids == holdout_ids | mates

    per_class: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in docs:
        per_class[d['class_id']][split_of[d['crop_id']]] += 1
    for class_id in range(len(CLASS_NAMES)):
        assert per_class[class_id]['train'] > 0, (class_id, dict(per_class[class_id]))
        assert per_class[class_id]['val'] > 0, (class_id, dict(per_class[class_id]))

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['group_key'] == 'image_id'
    assert manifest['split_counts'] == result.split_counts.to_dict()


@pytest.mark.asyncio
async def test_manifest_records_per_class_split_counts(tmp_path: Path) -> None:
    docs = _class_sized_cluster_docs()
    result = await _service(tmp_path, docs).export_dataset(seed=42, copy_images=False)

    manifest = json.loads(Path(result.manifest_path).read_text())
    rows = manifest['class_split_counts']
    assert [r['class_name'] for r in rows] == CLASS_NAMES
    assert [r['export_id'] for r in rows] == [0, 1, 2, 3, 4]
    assert [r['class_id'] for r in rows] == [0, 1, 2, 3, 4]

    split_of = _split_by_item(Path(result.export_dir), docs)
    for row in rows:
        expected: dict[str, int] = defaultdict(int)
        for d in docs:
            if d['class_id'] == row['class_id']:
                expected[split_of[d['crop_id']]] += 1
        assert (row['train'], row['val'], row['test']) == (
            expected['train'],
            expected['val'],
            expected['test'],
        )
    # class_split_counts count objects; image_count counts images (the
    # same-image mate shares its image with a holdout item).
    assert sum(r['train'] + r['val'] + r['test'] for r in rows) == result.object_count == len(docs)
    assert result.image_count == len({d['image_id'] for d in docs}) == len(docs) - 1
    assert manifest['split_object_counts'] == {
        s: sum(r[s] for r in rows) for s in ('train', 'val', 'test')
    }


# --------------------------------------------------------------- the splitter


def _row(item_id: str, image_id: str, class_id: int, *, holdout: bool = False) -> _ExportRow:
    return _ExportRow(
        item_id=item_id,
        image_id=image_id,
        image_path=f'{image_id}.jpg',
        bbox_norm=[0.0, 0.0, 1.0, 1.0],
        class_id=class_id,
        class_name=str(class_id),
        has_test_crop=holdout,
    )


def test_default_group_key_is_the_source_image() -> None:
    # 20 items of one class, one frozen holdout item, each on its own image:
    # only the holdout item's image goes to test.
    rows = [_row(f'i{i}', f'img{i}', 0, holdout=i == 0) for i in range(20)]
    splits = stratified_split(rows, seed=1, train_ratio=0.8, val_ratio=0.1)
    assert splits['i0'] == 'test'
    assert sorted(v for k, v in splits.items() if k != 'i0').count('test') == 0
    assert 'train' in splits.values()
    assert 'val' in splits.values()


def test_items_from_one_image_never_straddle_splits() -> None:
    rows = [
        _row(f'img{img}-{k}', f'img{img}', (img + k) % 3, holdout=img == 7)
        for img in range(40)
        for k in range(3)
    ]
    splits = stratified_split(rows, seed=9, train_ratio=0.8, val_ratio=0.1, group_key='image_id')
    by_image: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_image[r.image_id].add(splits[r.item_id])
    assert all(len(s) == 1 for s in by_image.values())
    # The holdout item's image-mates follow it to test.
    assert by_image['img7'] == {'test'}


def test_split_is_deterministic_and_seed_sensitive() -> None:
    rows = [_row(f'i{i}', f'img{i}', i % 4, holdout=i % 10 == 0) for i in range(120)]
    a = stratified_split(rows, seed=42, train_ratio=0.8, val_ratio=0.1)
    b = stratified_split(list(reversed(rows)), seed=42, train_ratio=0.8, val_ratio=0.1)
    c = stratified_split(rows, seed=43, train_ratio=0.8, val_ratio=0.1)
    assert a == b  # input order never matters
    assert a != c


def _class_splits(splits: dict[str, str], rows: list[_ExportRow], class_id: int) -> list[str]:
    return sorted(splits[r.item_id] for r in rows if r.class_id == class_id)


@pytest.mark.parametrize(
    ('n_groups', 'expected'),
    [
        (1, ['train']),
        (2, ['train', 'val']),
        (3, ['train', 'train', 'val']),
        (4, ['train', 'train', 'train', 'val']),
    ],
)
def test_small_class_with_a_holdout_splits_remaining_groups_train_val(
    n_groups: int, expected: list[str]
) -> None:
    rows = [_row('held', 'img-held', 0, holdout=True)]
    rows += [_row(f'i{i}', f'img{i}', 0) for i in range(n_groups)]
    splits = stratified_split(rows, seed=5, train_ratio=0.8, val_ratio=0.1)
    assert splits['held'] == 'test'
    assert sorted(splits[r.item_id] for r in rows if not r.has_test_crop) == expected


@pytest.mark.parametrize(
    ('n_groups', 'expected'),
    [
        (1, ['train']),
        (2, ['train', 'val']),
        (3, ['test', 'train', 'val']),
        (5, ['test', 'train', 'train', 'train', 'val']),
    ],
)
def test_small_class_without_a_holdout_gets_each_split_in_priority_order(
    n_groups: int, expected: list[str]
) -> None:
    rows = [_row(f'i{i}', f'img{i}', 0) for i in range(n_groups)]
    splits = stratified_split(rows, seed=5, train_ratio=0.8, val_ratio=0.1)
    assert _class_splits(splits, rows, 0) == expected


def test_holdout_only_class_contributes_only_to_test() -> None:
    rows = [_row('held', 'img-held', 3, holdout=True)]
    rows += [_row(f'i{i}', f'img{i}', 0) for i in range(10)]
    splits = stratified_split(rows, seed=5, train_ratio=0.8, val_ratio=0.1)
    assert splits['held'] == 'test'
    assert 'train' in _class_splits(splits, rows, 0)


def test_zero_ratio_split_is_never_forced() -> None:
    """The single-class exporter's skip_test_split path passes a
    train/val-only ratio; the >=1-per-split minimum must not invent a
    test group there."""
    rows = [_row(f'i{i}', f'img{i}', 0) for i in range(3)]
    splits = stratified_split(rows, seed=5, train_ratio=0.89, val_ratio=0.11)
    assert 'test' not in splits.values()


# ----------------------------------------------------------- /export/status


@pytest.fixture
def status_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = AsyncMock
    return TestClient(app)


@pytest.mark.asyncio
async def test_export_status_serves_the_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_client: TestClient
) -> None:
    service = _service(tmp_path, _class_sized_cluster_docs())
    result = await service.export_dataset(version_tag='v-test', seed=42, copy_images=False)
    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        lambda: resolve_current_export_dir(service.config),
    )

    body = status_client.get('/curation/export/status').json()

    assert body['status'] == 'success'
    assert body['path'] == str(Path(result.export_dir).resolve())
    assert body['export_dir'] == body['path']
    assert body['version_tag'] == 'v-test'
    assert body['dataset_sha'] == result.dataset_sha
    assert body['image_count'] == result.image_count
    assert body['group_key'] == 'image_id'
    assert body['seed'] == 42
    assert body['split_counts'] == result.split_counts.to_dict()
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert body['class_split_counts'] == manifest['class_split_counts']
    assert body['last_run'] == manifest['finished_at']
    assert body['skipped_items'] == manifest['skipped_items']
    assert body['skipped_items'] is not None


def test_export_status_skipped_items_is_null_for_a_legacy_manifest_without_the_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_client: TestClient
) -> None:
    """An export written before ``skipped_items`` existed must report
    ``null``, not a fabricated zero and not a validation error."""
    export_dir = tmp_path / 'legacy-export'
    export_dir.mkdir()
    (export_dir / 'manifest.json').write_text(json.dumps({'version_tag': 'pre-skipped-items'}))
    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        lambda: export_dir,
    )

    body = status_client.get('/curation/export/status').json()

    assert body['status'] == 'success'
    assert body['skipped_items'] is None


def test_export_status_is_idle_without_an_export(
    monkeypatch: pytest.MonkeyPatch, status_client: TestClient
) -> None:
    def _missing() -> Path:
        raise FileNotFoundError

    monkeypatch.setattr('src.routers.curation.export._resolve_current_export_dir', _missing)
    body = status_client.get('/curation/export/status').json()
    assert body['status'] == 'idle'
    assert body['path'] is None
    assert body['split_counts'] is None
    assert body['class_split_counts'] is None


def test_export_status_is_in_the_openapi_contract(status_client: TestClient) -> None:
    spec = status_client.get('/openapi.json').json()
    op = spec['paths']['/curation/export/status']['get']
    ref = op['responses']['200']['content']['application/json']['schema']['$ref']
    schema = spec['components']['schemas'][ref.rsplit('/', 1)[-1]]
    for key in ('path', 'version_tag', 'dataset_sha', 'split_counts', 'class_split_counts'):
        assert key in schema['properties']

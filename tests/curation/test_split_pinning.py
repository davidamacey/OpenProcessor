"""W10.9: the export honours the split a dataset import filed each frame
under, the frozen holdout always wins, and ``recompute`` ignores pins."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.op_export_fixtures import (
    export_multi_class,
    export_region_whole_frame,
    item,
    multi_registry,
    region_item,
)
from src.config.region_state import RegionStatus
from src.services.curation.export_split import (
    pins_from_rows,
    stratified_split,
    stratified_split_with_stats,
)
from src.services.curation.export_support import (
    frozen_sha_of_names,
    frozen_test_sha_of,
    source_frozen_test_sha,
)


if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Row:
    item_id: str
    class_id: int
    image_id: str
    has_test_crop: bool = False
    dataset_split: str | None = None


def _rows(n: int = 30) -> list[Row]:
    return [Row(item_id=f'i{k}', class_id=k % 2, image_id=f'img{k}') for k in range(n)]


def _run(rows, pins=None):
    return stratified_split_with_stats(
        rows, seed=7, train_ratio=0.6, val_ratio=0.2, group_key='image_id', pinned=pins
    )


def test_pinned_groups_keep_their_split_and_the_rest_follow_the_rule() -> None:
    rows = _rows()
    baseline, _ = _run(rows)
    pins = {'i0': 'test', 'i1': 'test', 'i2': 'val', 'i3': 'train'}
    splits, stats = _run(rows, pins)
    assert [splits[k] for k in ('i0', 'i1', 'i2', 'i3')] == ['test', 'test', 'val', 'train']
    assert stats.pinned_groups == 4
    assert stats.computed_groups == 26
    assert stats.pinned_split_conflicts == 0
    assert stats.pinned_overridden_by_holdout == 0
    # Unpinned rows are still allocated across every split.
    assert {splits[r.item_id] for r in rows if r.item_id not in pins} == {'train', 'val', 'test'}
    # Pinning nothing reproduces the unpinned result exactly.
    assert _run(rows, {})[0] == baseline


def test_the_holdout_overrides_a_pin_and_is_counted() -> None:
    rows = _rows()
    rows[0].has_test_crop = True
    rows[1].has_test_crop = True
    splits, stats = _run(rows, {'i0': 'train', 'i1': 'test', 'i5': 'val'})
    assert splits['i0'] == 'test'
    assert splits['i1'] == 'test'
    assert splits['i5'] == 'val'
    assert stats.pinned_overridden_by_holdout == 1
    assert stats.pinned_groups == 1


def test_conflicting_members_take_the_most_common_split_ties_test_val_train() -> None:
    rows = [Row('a', 0, 'g'), Row('b', 0, 'g'), Row('c', 0, 'g')] + [
        Row(f'o{k}', 0, f'o{k}') for k in range(10)
    ]
    splits, stats = _run(rows, {'a': 'train', 'b': 'val', 'c': 'val'})
    assert splits['a'] == splits['b'] == splits['c'] == 'val'
    assert stats.pinned_split_conflicts == 1
    tie, tie_stats = _run(rows, {'a': 'train', 'b': 'val', 'c': 'test'})
    assert tie['a'] == 'test'
    assert tie_stats.pinned_split_conflicts == 1
    low, _ = _run(rows, {'a': 'train', 'b': 'val'})
    assert low['a'] == 'val'


def test_invalid_pin_values_are_ignored() -> None:
    rows = _rows(10)
    splits, stats = _run(rows, {'i0': 'bogus', 'i1': ''})
    assert stats.pinned_groups == 0
    assert splits == _run(rows)[0]


def test_stratified_split_wrapper_returns_only_the_assignment() -> None:
    rows = _rows(12)
    assert (
        stratified_split(
            rows,
            seed=7,
            train_ratio=0.6,
            val_ratio=0.2,
            group_key='image_id',
            pinned={'i0': 'test'},
        )['i0']
        == 'test'
    )


def test_pins_from_rows_honours_split_mode() -> None:
    rows = [Row('a', 0, 'g', dataset_split='val'), Row('b', 0, 'h', dataset_split=None)]
    assert pins_from_rows(rows, 'keep_imported') == {'a': 'val'}
    assert pins_from_rows(rows, 'recompute') is None


def test_source_frozen_sha_matches_the_name_digest_of_the_source_export(tmp_path: Path) -> None:
    labels = tmp_path / 'labels' / 'test'
    labels.mkdir(parents=True)
    for stem in ('s1', 's2'):
        (labels / f'{stem}.txt').write_text('')
    on_disk = frozen_test_sha_of(tmp_path)
    assert on_disk == frozen_sha_of_names(['s1.txt', 's2.txt'])
    # The re-exported frames carry new ids but remember the source stems.
    assert source_frozen_test_sha(['s2', 's1'], ['new-a', 'new-b']) == on_disk
    assert source_frozen_test_sha([None, None], ['x', 'y']) == ''
    assert frozen_sha_of_names([]) == ''
    # A test row that was never imported contributes its own stem.
    assert source_frozen_test_sha(['s1', None], ['n1', 'own']) == frozen_sha_of_names(
        ['s1.txt', 'own.txt']
    )


# ------------------------------------------------------------- exporters


async def _export_with_splits(tmp_path: Path, **kwargs) -> dict:
    reg = multi_registry(tmp_path / 'reg', ['car', 'truck'])
    docs = {}
    for k in range(12):
        docs[f'c{k}'] = item(
            f'c{k}',
            f'img{k}',
            k % 2,
            ['car', 'truck'][k % 2],
            [0.1, 0.1, 0.5, 0.5],
            dataset_split='test' if k < 3 else 'val',
            import_source_stem=f'src{k}',
        )
    export_dir, _ = await export_multi_class(tmp_path, docs, reg, **kwargs)
    manifest = json.loads((export_dir / 'manifest.json').read_text())
    placed = {p.stem: p.parent.name for p in export_dir.glob('labels/*/*.txt')}
    return {'manifest': manifest, 'placed': placed}


@pytest.mark.asyncio
async def test_multi_class_export_keeps_imported_splits(tmp_path: Path) -> None:
    out = await _export_with_splits(tmp_path)
    expected = {f'img{k}': 'test' if k < 3 else 'val' for k in range(12)}
    assert out['placed'] == expected
    m = out['manifest']
    assert m['split_mode'] == 'keep_imported'
    assert m['pinned_groups'] == 12
    assert m['computed_groups'] == 0
    # The test rows' source stems reproduce the source export's identity.
    assert m['source_frozen_test_sha'] == frozen_sha_of_names(f'src{k}.txt' for k in range(3))


@pytest.mark.asyncio
async def test_multi_class_recompute_ignores_the_stored_split(tmp_path: Path) -> None:
    out = await _export_with_splits(tmp_path, split_mode='recompute')
    expected = {f'img{k}': 'test' if k < 3 else 'val' for k in range(12)}
    assert out['placed'] != expected
    assert out['manifest']['split_mode'] == 'recompute'
    assert out['manifest']['pinned_groups'] == 0


@pytest.mark.asyncio
async def test_single_class_export_keeps_imported_splits_and_records_stems(
    tmp_path: Path,
) -> None:
    docs = {
        f'r{k}': region_item(
            f'r{k}',
            f'frame{k}',
            RegionStatus.DETECTED,
            box=[0.2, 0.2, 0.4, 0.4],
            dataset_split='test' if k < 2 else 'train',
            import_source_stem=f'orig{k}',
        )
        for k in range(8)
    }
    export_dir, _ = await export_region_whole_frame(tmp_path, docs)
    placed = {p.stem: p.parent.name for p in export_dir.glob('labels/*/*.txt')}
    assert placed == {f'frame{k}': 'test' if k < 2 else 'train' for k in range(8)}
    manifest = json.loads((export_dir / 'manifest.json').read_text())
    assert manifest['split_mode'] == 'keep_imported'
    assert manifest['pinned_groups'] == 8
    assert manifest['source_frozen_test_sha'] == frozen_sha_of_names(['orig0.txt', 'orig1.txt'])

    export_dir2, _ = await export_region_whole_frame(
        tmp_path / 'again', docs, split_mode='recompute'
    )
    placed2 = {p.stem: p.parent.name for p in export_dir2.glob('labels/*/*.txt')}
    assert placed2 != placed
    manifest2 = json.loads((export_dir2 / 'manifest.json').read_text())
    assert manifest2['split_mode'] == 'recompute'
    assert 'source_frozen_test_sha' not in manifest2 or manifest2['pinned_groups'] == 0


@pytest.mark.asyncio
async def test_exports_without_imports_are_unchanged(tmp_path: Path) -> None:
    reg = multi_registry(tmp_path / 'reg', ['car'])
    docs = {f'c{k}': item(f'c{k}', f'img{k}', 0, 'car', [0.1, 0.1, 0.5, 0.5]) for k in range(10)}
    export_dir, _ = await export_multi_class(tmp_path, docs, reg)
    manifest = json.loads((export_dir / 'manifest.json').read_text())
    assert manifest['pinned_groups'] == 0
    assert manifest['negative_images'] == 0
    assert 'source_frozen_test_sha' not in manifest

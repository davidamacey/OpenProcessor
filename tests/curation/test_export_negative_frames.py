"""W10.8: imported reviewed-negative frames reach both exporters as
background training data, with the coverage rule and provenance intact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from curation.dataset_import.op_export_fixtures import (
    IMAGES,
    ITEMS,
    config,
    export_multi_class,
    export_region_whole_frame,
    item,
    multi_registry,
    region_item,
)
from curation.query_fakes import QueryFakeOpenSearch
from src.config.region_state import RegionStatus
from src.services.curation.export_single_class import (
    SingleClassExportProfile,
    SingleClassExportService,
)


pytestmark = pytest.mark.asyncio


def _neg(image_id: str, negative_for: list[str], **extra: Any) -> dict[str, Any]:
    return {
        'image_id': image_id,
        'image_path': f'/data/{image_id}.jpg',
        'import_label_state': 'negative',
        'negative_for': negative_for,
        **extra,
    }


def _docs() -> dict[str, dict[str, Any]]:
    docs = {}
    for k in range(10):
        cls, name = [(0, 'car'), (1, 'truck')][k % 2]
        docs[f'c{k}'] = item(f'c{k}', f'img{k}', cls, name, [0.1, 0.1, 0.5, 0.5])
    return docs


def _placed(export_dir: Path) -> dict[str, str]:
    return {p.stem: p.parent.name for p in export_dir.glob('labels/*/*.txt')}


async def _multi(tmp_path: Path, images: dict, **kwargs: Any):
    reg = multi_registry(tmp_path / 'reg', ['car', 'truck'])
    export_dir, _ = await export_multi_class(tmp_path, _docs(), reg, images=images, **kwargs)
    return export_dir, json.loads((export_dir / 'manifest.json').read_text())


async def test_multi_class_writes_a_negative_only_when_it_covers_every_exported_class(
    tmp_path: Path,
) -> None:
    images = {
        'covers': _neg('covers', ['car', 'truck'], dataset_split='val'),
        'partial': _neg('partial', ['car']),
        'alsoCovers': _neg('alsoCovers', ['truck', 'car', 'bus'], dataset_split='train'),
    }
    export_dir, manifest = await _multi(tmp_path, images)
    placed = _placed(export_dir)
    assert placed['covers'] == 'val'
    assert placed['alsoCovers'] == 'train'
    assert 'partial' not in placed
    for stem in ('covers', 'alsoCovers'):
        label = export_dir / 'labels' / placed[stem] / f'{stem}.txt'
        assert label.read_text() == ''
    assert manifest['negative_images'] == 2
    assert manifest['negative_frames_skipped_partial'] == 1
    assert manifest['image_count'] == 12
    assert sum(manifest['split_counts'].values()) == 12
    assert manifest['object_count'] == 10


async def test_multi_class_include_negative_frames_false_writes_none(tmp_path: Path) -> None:
    images = {'covers': _neg('covers', ['car', 'truck'])}
    export_dir, manifest = await _multi(tmp_path, images, include_negative_frames=False)
    assert 'covers' not in _placed(export_dir)
    assert manifest['negative_images'] == 0


async def test_multi_class_negative_on_an_exported_image_is_not_duplicated(
    tmp_path: Path,
) -> None:
    images = {'img3': _neg('img3', ['car', 'truck'])}
    export_dir, manifest = await _multi(tmp_path, images)
    label = next(export_dir.glob('labels/*/img3.txt'))
    assert label.read_text().strip()  # still carries its object
    assert manifest['negative_images'] == 0


async def test_multi_class_negative_split_follows_split_mode(tmp_path: Path) -> None:
    images = {f'n{k}': _neg(f'n{k}', ['car', 'truck'], dataset_split='test') for k in range(8)}
    keep_dir, _ = await _multi(tmp_path / 'keep', images)
    assert {_placed(keep_dir)[f'n{k}'] for k in range(8)} == {'test'}
    recomputed_dir, _ = await _multi(tmp_path / 'rec', images, split_mode='recompute')
    assert {_placed(recomputed_dir)[f'n{k}'] for k in range(8)} != {'test'}


async def test_negative_scan_ignores_docs_that_are_not_marked_negative(tmp_path: Path) -> None:
    images = {'x': {'image_id': 'x', 'image_path': '/d/x.jpg', 'negative_for': ['car', 'truck']}}
    export_dir, manifest = await _multi(tmp_path, images)
    assert 'x' not in _placed(export_dir)
    assert manifest['negative_images'] == 0
    assert manifest['negative_frames_skipped_partial'] == 0


# ---------------------------------------------------------------- single class


async def _item_mode(tmp_path: Path, images: dict, **export_kwargs: Any):
    reg = multi_registry(tmp_path / 'reg', ['car', 'truck'])
    fake = QueryFakeOpenSearch({ITEMS: _docs(), IMAGES: images})
    service = SingleClassExportService(
        fake,
        profile=SingleClassExportProfile(name='cars', class_ids=(0,), box_source='item'),
        config=config(tmp_path),
        registry=reg,
    )
    result = await service.export(version_tag='fx', copy_images=False, **export_kwargs)
    export_dir = Path(result.export_dir)
    return export_dir, json.loads((export_dir / 'manifest.json').read_text())


async def test_single_class_item_mode_negative_needs_an_exported_class_name(
    tmp_path: Path,
) -> None:
    images = {
        'forCar': _neg('forCar', ['car'], import_stratum='bg:imp'),
        'forTruck': _neg('forTruck', ['truck']),
    }
    export_dir, manifest = await _item_mode(tmp_path, images, empty_bg_ratio=100.0)
    placed = _placed(export_dir)
    assert 'forCar' in placed
    assert 'forTruck' not in placed
    assert (export_dir / 'labels' / placed['forCar'] / 'forCar.txt').read_text() == ''
    strata = json.loads((export_dir / 'stratum_map.json').read_text())
    assert strata['forCar'] == 'bg:imp'
    assert manifest['background_images'] >= 1


async def test_single_class_hard_negative_frames_are_kept_in_full(tmp_path: Path) -> None:
    images = {
        'soft': _neg('soft', ['car']),
        'hard': _neg('hard', ['car'], import_hard_negative=True, import_stratum='neg:9'),
    }
    export_dir, manifest = await _item_mode(tmp_path, images, empty_bg_ratio=0.0)
    placed = _placed(export_dir)
    assert 'hard' in placed
    assert 'soft' not in placed
    assert manifest['false_positive_background_images'] >= 1
    assert json.loads((export_dir / 'stratum_map.json').read_text())['hard'] == 'neg:9'


async def test_single_class_region_negative_matches_the_region_class_name(
    tmp_path: Path,
) -> None:
    docs = {
        'r0': region_item('r0', 'frameA', RegionStatus.DETECTED, box=[0.2, 0.2, 0.4, 0.4]),
    }
    images = {
        'plateNeg': _neg('plateNeg', ['plate']),
        'carNeg': _neg('carNeg', ['car']),
    }
    export_dir, _ = await export_region_whole_frame(tmp_path, docs, images=images)
    placed = _placed(export_dir)
    assert 'plateNeg' in placed
    assert 'carNeg' not in placed


async def test_import_stratum_on_an_items_frame_is_preferred_for_a_positive(
    tmp_path: Path,
) -> None:
    docs = {
        'r0': region_item(
            'r0',
            'frameA',
            RegionStatus.DETECTED,
            box=[0.2, 0.2, 0.4, 0.4],
            import_stratum='pos:source-7',
        ),
        'r1': region_item('r1', 'frameB', RegionStatus.DETECTED, box=[0.2, 0.2, 0.4, 0.4]),
    }
    export_dir, _ = await export_region_whole_frame(tmp_path, docs)
    strata = json.loads((export_dir / 'stratum_map.json').read_text())
    assert strata['frameA'] == 'pos:source-7'
    assert strata['frameB'] != 'pos:source-7'


async def test_item_crop_mode_adds_no_frame_sized_backgrounds(tmp_path: Path) -> None:
    docs = {
        'r0': region_item('r0', 'frameA', RegionStatus.DETECTED, box=[0.2, 0.2, 0.4, 0.4]),
    }
    export_dir, _ = await export_region_whole_frame(
        tmp_path, docs, images={'plateNeg': _neg('plateNeg', ['plate'])}, image_mode='item_crop'
    )
    assert 'plateNeg' not in _placed(export_dir)


class _IgnoresTheQuery:
    """A fake that returns the same hits whatever it is asked, like a real
    index would if the server-side term filter were missing or wrong."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    async def search(self, **_kw: Any) -> dict[str, Any]:
        hits = [{'_id': d.get('image_id'), '_source': d} for d in self.docs]
        return {'_scroll_id': 's', 'hits': {'hits': hits}}

    async def scroll(self, **_kw: Any) -> dict[str, Any]:
        return {'_scroll_id': 's', 'hits': {'hits': []}}

    async def clear_scroll(self, **_kw: Any) -> dict[str, Any]:
        return {}


async def test_the_negative_scan_re_checks_the_state_client_side() -> None:
    from src.services.curation.export_negatives import scroll_negative_frames

    fake = _IgnoresTheQuery(
        [
            _neg('yes', ['car']),
            {'image_id': 'no', 'image_path': '/d/no.jpg', 'negative_for': ['car']},
            {**_neg('labeled', ['car']), 'import_label_state': 'labeled'},
        ]
    )
    frames = await scroll_negative_frames(fake, images_index=IMAGES)
    assert [f.image_id for f in frames] == ['yes']

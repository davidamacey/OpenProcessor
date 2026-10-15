"""Characterization of :meth:`GenericYoloExportService.export_dataset`.

A fixed corpus (two registered classes, an unregistered class, a frame with
an unlabeled object, a frozen test-holdout frame, imported splits, skipped
hits and reviewed-negative frames) is exported under two option sets. Every
file the export writes, the result envelope and the manifest are compared
with a golden snapshot written from the code before ``export_dataset`` was
decomposed, so a refactor must leave the written dataset byte-identical.
Regenerate deliberately with ``OP_REGEN_EXPORT_GOLDEN=1``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import pytest

from curation.export_fakes import make_registry
from src.config import CurationConfig
from src.services.curation.export import GenericYoloExportService


GOLDEN = Path(__file__).parent / 'golden' / 'export_dataset.json'
VOLATILE = {'started_at', 'finished_at', 'exported_at'}


def _item(idx: int, image: str, class_id: int, **extra: Any) -> dict[str, Any]:
    box = [0.05 + 0.02 * (idx % 7), 0.1, 0.45 + 0.02 * (idx % 5), 0.6]
    return {
        'crop_id': f'crop-{idx}',
        'image_id': image,
        'image_path': f'{image}.jpg',
        'bbox_norm': box,
        'class_id': class_id,
        'class_name': {0: 'car', 1: 'truck', 2: 'bus'}[class_id],
        **extra,
    }


def _items() -> list[dict[str, Any]]:
    items = [
        _item(1, 'img-1', 0, test_holdout=True, import_source_stem='src-1'),
        _item(2, 'img-1', 2),  # unregistered class on an exported frame -> unlabeled
        _item(3, 'img-2', 1, dataset_split='val'),
        _item(4, 'img-3', 2),  # only an unregistered class -> dropped
        {**_item(5, 'img-4', 0), 'image_id': ''},  # skipped: no image id
        {**_item(6, 'img-5', 0), 'bbox_norm': None},  # skipped: no usable box
    ]
    items.extend(
        _item(n, f'img-{n}', n % 2, dataset_split=('train' if n % 5 == 0 else None))
        for n in range(7, 27)
    )
    items.append(_item(30, 'img-12', 1))  # second object on an existing frame
    return items


def _negatives() -> list[dict[str, Any]]:
    return [
        {
            'image_id': 'neg-1',
            'image_path': 'n1.jpg',
            'import_label_state': 'negative',
            'negative_for': ['car', 'truck'],
            'dataset_split': 'val',
        },
        {
            'image_id': 'neg-2',
            'image_path': 'n2.jpg',
            'import_label_state': 'negative',
            'negative_for': ['car'],
        },
        {
            'image_id': 'neg-3',
            'image_path': 'n3.jpg',
            'import_label_state': 'negative',
            'negative_for': ['car', 'truck', 'bus'],
            'import_source_stem': 'neg-src',
        },
    ]


class _IndexFake:
    """Serves one page per search, picked by index; no scroll continuation."""

    def __init__(self, by_index: dict[str, list[dict[str, Any]]]) -> None:
        self.by_index = by_index

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        docs = self.by_index.get(index, [])
        hits = [{'_id': d.get('crop_id') or d.get('image_id', ''), '_source': d} for d in docs]
        return {'_scroll_id': 's', 'hits': {'hits': hits}}

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None


CASES: dict[str, dict[str, Any]] = {
    'defaults': {'version_tag': 'golden', 'seed': 11},
    'capped_fully_labeled_recompute': {
        'version_tag': 'golden-2',
        'seed': 5,
        'max_images': 9,
        'require_fully_labeled_images': True,
        'split_mode': 'recompute',
        'include_negative_frames': False,
    },
}


def _normalize(value: Any, export_dir: str) -> Any:
    if isinstance(value, dict):
        return {
            k: ('<time>' if k in VOLATILE else _normalize(v, export_dir)) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_normalize(v, export_dir) for v in value]
    if isinstance(value, str):
        return value.replace(export_dir, '<TMP>')
    return value


async def _run(tmp_path: Path, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    cfg = CurationConfig(export_root=tmp_path / name / 'exports')
    fake = _IndexFake({cfg.items_index: _items(), cfg.images_index: _negatives()})
    service = GenericYoloExportService(
        fake, config=cfg, registry=make_registry(tmp_path / name / 'registry', ['car', 'truck'])
    )
    out_dir = tmp_path / name / 'dataset'
    result = await service.export_dataset(export_dir=out_dir, copy_images=False, **kwargs)
    export_dir = str(tmp_path / name)
    files = {
        str(p.relative_to(out_dir)): _normalize(p.read_text(), export_dir)
        for p in sorted(out_dir.rglob('*'))
        if p.is_file()
    }
    for fname in ('manifest.json', 'class_registry.json'):
        files[fname] = _normalize(json.loads(files[fname]), export_dir)
    return {
        'files': files,
        'result': _normalize(dataclasses.asdict(result), export_dir),
        'current_link_target': _normalize(
            str(Path(cfg.export_root / 'current').resolve()), export_dir
        ),
    }


@pytest.mark.asyncio
async def test_export_dataset_writes_the_golden_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_BUILD_SHA', 'golden-sha')
    snap = {name: await _run(tmp_path, name, kw) for name, kw in CASES.items()}
    snap = json.loads(json.dumps(snap))
    if os.environ.get('OP_REGEN_EXPORT_GOLDEN') == '1':
        GOLDEN.write_text(json.dumps(snap, indent=2, sort_keys=True) + '\n')
        pytest.skip('golden regenerated')
    assert snap == json.loads(GOLDEN.read_text())


@pytest.mark.asyncio
async def test_the_golden_corpus_reaches_the_interesting_export_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_BUILD_SHA', 'golden-sha')
    default = (await _run(tmp_path, 'a', CASES['defaults']))['files']['manifest.json']
    assert default['dropped_unregistered_class_ids'] == {'2': 2}
    assert default['skipped_items'] == {'no_image_id': 1, 'no_usable_box_or_class': 1}
    assert default['unlabeled_items_on_exported_images'] == 1
    assert default['negative_images'] == 2
    assert default['negative_frames_skipped_partial'] == 1
    assert default['frozen_holdout_sha'] is not None
    assert default['split_counts']['test'] >= 1
    capped = (await _run(tmp_path, 'b', CASES['capped_fully_labeled_recompute']))['files'][
        'manifest.json'
    ]
    assert capped['sampling_mode'] == 'stratified_even'
    assert capped['images_dropped_not_fully_labeled'] == 1
    assert capped['negative_images'] == 0

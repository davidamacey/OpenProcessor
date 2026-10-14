"""Generic export: even stratified sampling and the max-images cap."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from curation.export_fakes import generic_service
from src.services.curation.export import even_stratified_sample


@pytest.mark.asyncio
async def test_export_dataset_respects_max_images(tmp_path):
    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_id': f'img-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': 0,
            'class_name': 'car',
        }
        for i in range(5)
    ]
    service = generic_service(tmp_path, docs, ['car'])

    result = await service.export_dataset(max_images=2, copy_images=False)

    assert result.image_count == 2


# ---------------------------------------------------------------------------
# Class-balanced sampling under a max_images cap (G10)
# ---------------------------------------------------------------------------


def test_even_stratified_sample_keeps_every_stratum():
    """A cap >= the stratum count must leave no stratum empty, even when
    one stratum dominates the input and sorts first."""
    rows = [('a', i) for i in range(100)] + [('b', 0), ('b', 1)] + [('c', 0)]
    picked = even_stratified_sample(rows, 9, lambda r: r[0], random.Random(1))

    assert len(picked) == 9
    assert {r[0] for r in picked} == {'a', 'b', 'c'}
    # Even, not proportional: 'b' and 'c' are exhausted rather than sampled
    # in proportion to their 2/103 and 1/103 share.
    assert sum(1 for r in picked if r[0] == 'b') == 2
    assert sum(1 for r in picked if r[0] == 'c') == 1


def test_even_stratified_sample_passthrough_and_edges():
    rows = list(range(5))
    assert even_stratified_sample(rows, None, str, random.Random(0)) == rows
    assert even_stratified_sample(rows, 10, str, random.Random(0)) == rows
    assert even_stratified_sample(rows, 0, str, random.Random(0)) == []


def test_even_stratified_sample_is_seed_reproducible():
    rows = [(f'cls-{i % 4}', i) for i in range(40)]
    a = even_stratified_sample(rows, 11, lambda r: r[0], random.Random(7))
    b = even_stratified_sample(rows, 11, lambda r: r[0], random.Random(7))
    assert a == b


def _imbalanced_docs() -> list[dict[str, Any]]:
    """40 cars, 3 trucks, 1 bus -- cars sort first, as a scroll would return
    them, so plain ``hits[:n]`` truncation drops truck and bus entirely."""
    spec = [('car', 0, 40), ('truck', 1, 3), ('bus', 2, 1)]
    return [
        {
            'crop_id': f'{name}-{i}',
            'image_id': f'{name}-img-{i}',
            'image_path': f'{name}-{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': class_id,
            'class_name': name,
        }
        for name, class_id, count in spec
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_max_images_cap_keeps_rare_classes_and_hits_the_cap(tmp_path):
    """G10: the cap is an even per-class sample applied after dedup, not a
    truncation of the raw hit list."""
    service = generic_service(tmp_path, _imbalanced_docs(), ['car', 'truck', 'bus'])

    result = await service.export_dataset(seed=42, max_images=12, copy_images=False)

    assert result.image_count == 12  # exactly the cap, not some count below it

    label_stats = json.loads((Path(result.export_dir) / 'label_stats.json').read_text())
    # (a) no class silently vanishes...
    assert label_stats['car'] > 0
    assert label_stats['truck'] == 3  # small class taken whole
    assert label_stats['bus'] == 1
    # ...and (b) the dominant class is the one trimmed to fit the budget.
    assert label_stats['car'] == 8
    assert sum(label_stats.values()) == 12

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['sampling_mode'] == 'stratified_even'
    assert manifest['max_images'] == 12
    assert manifest['image_count'] == 12


@pytest.mark.asyncio
async def test_uncapped_export_records_sampling_mode_all(tmp_path):
    service = generic_service(tmp_path, _imbalanced_docs(), ['car', 'truck', 'bus'])

    result = await service.export_dataset(seed=42, copy_images=False)

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['sampling_mode'] == 'all'
    assert manifest['max_images'] is None
    assert result.image_count == 44


@pytest.mark.asyncio
async def test_capped_export_is_reproducible_from_the_recorded_seed(tmp_path):
    """The sample is RNG-driven, so pin that the recorded seed reproduces it."""
    docs = _imbalanced_docs()
    service_a = generic_service(tmp_path / 'a', docs, ['car', 'truck', 'bus'])
    service_b = generic_service(tmp_path / 'b', docs, ['car', 'truck', 'bus'])

    result_a = await service_a.export_dataset(seed=5, max_images=12, copy_images=False)
    result_b = await service_b.export_dataset(seed=5, max_images=12, copy_images=False)

    assert result_a.dataset_sha == result_b.dataset_sha
    assert result_a.split_counts.to_dict() == result_b.split_counts.to_dict()

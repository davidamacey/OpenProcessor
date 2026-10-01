"""Frozen test splits across a combine (owner decision D6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.curation.holdout import compute_holdout_sha, select_test_holdout
from src.services.projects.combine import service

from .world import World, run_job


BOX = [0.1, 0.1, 0.5, 0.5]
BOX2 = [0.6, 0.6, 0.9, 0.9]
MAPPING = {
    'cars-a': [{'dataset_class': 'car', 'action': 'create', 'new_class_name': 'car'}],
    'cars-b': [{'dataset_class': 'car', 'action': 'map', 'new_class_name': 'car'}],
}
SHARED = 900


def build(world: World) -> dict[str, str]:
    """A: a train frame, a frame filed under the ``test`` split, and a frame B
    also has. B: its copy of the shared frame is frozen test, plus a train
    frame."""
    world.project('cars-a', ['car'])
    world.project('cars-b', ['car'])
    validated = {'class_validated': True, 'class_source': 'human', 'label_source': 'human'}
    ids = {}
    ids['a_train'], _ = world.add_image(
        'cars-a', items=[{'cls': 'car', 'bbox': BOX, **validated}], split='train'
    )
    ids['a_test'], _ = world.add_image(
        'cars-a', items=[{'cls': 'car', 'bbox': BOX, **validated}], split='test'
    )
    ids['a_frozen'], _ = world.add_image(
        'cars-a', items=[{'cls': 'car', 'bbox': BOX, 'test_holdout': True, **validated}]
    )
    # A guessed the shared frame's box; B's human label wins the merge.
    ids['a_shared'], _ = world.add_image(
        'cars-a', seed=SHARED, items=[{'cls': 'car', 'bbox': BOX}], split='train'
    )
    ids['b_shared'], _ = world.add_image(
        'cars-b',
        seed=SHARED,
        items=[{'cls': 'car', 'bbox': BOX, 'test_holdout': True, **validated}],
    )
    ids['b_train'], _ = world.add_image('cars-b', items=[{'cls': 'car', 'bbox': BOX, **validated}])
    return ids


def by_origin(world: World) -> dict[str, dict]:
    return {d['origin_image_id']: d for d in world.images('combined').values()}


def holdout_images(world: World) -> set[str]:
    return {d['image_id'] for d in world.items('combined').values() if d.get('test_holdout')}


@pytest.mark.asyncio
async def test_preserve_union_keeps_anything_test_in_any_source(world: World) -> None:
    ids = build(world)
    store, target = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    images = by_origin(world)
    expect_test = {
        images[ids['a_test']]['image_id'],  # the W10 test split
        images[ids['a_frozen']]['image_id'],  # frozen in A itself
        images[ids['a_shared']]['image_id'],  # B froze its copy, A's was train
    }
    assert holdout_images(world) == expect_test
    # Every item of a test image is test, including the merged non-priority copy.
    for doc in world.items('combined').values():
        assert bool(doc['test_holdout']) == (doc['image_id'] in expect_test)
    # Nothing a source froze can land in the target's train set.
    frozen_in_sources = {ids['a_test'], ids['a_frozen'], ids['a_shared'], ids['b_shared']}
    for image_id, doc in world.images('combined').items():
        if image_id not in expect_test:
            assert doc['origin_image_id'] not in frozen_in_sources

    freeze_dir = target.resources.project_state_dir / 'test_holdout' / 'imports'
    (record_path,) = list(Path(freeze_dir).glob('*.json'))
    record = json.loads(record_path.read_text())
    frozen = sorted(d['crop_id'] for d in world.items('combined').values() if d['test_holdout'])
    assert record['crop_ids'] == frozen
    assert record['test_holdout_sha'] == compute_holdout_sha(frozen)
    assert store.job.read()['report']['holdout_items'] == len(frozen)


@pytest.mark.asyncio
async def test_the_non_priority_copy_decides_when_the_priority_copy_was_train(
    world: World,
) -> None:
    ids = build(world)
    await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    shared = by_origin(world)[ids['a_shared']]['image_id']
    assert all(
        d['test_holdout'] for d in world.items('combined').values() if d['image_id'] == shared
    )
    # Reverse the priority: B first, its copy is the one that is copied.
    swapped = {'cars-b': MAPPING['cars-b'], 'cars-a': MAPPING['cars-a']}
    await run_job(world, world.request(['cars-b', 'cars-a'], swapped, target='flipped'))
    flipped = {d['origin_image_id']: d for d in world.images('flipped').values()}
    frozen = {d['image_id'] for d in world.items('flipped').values() if d['test_holdout']}
    assert frozen == {flipped[ids[k]]['image_id'] for k in ('b_shared', 'a_test', 'a_frozen')}


@pytest.mark.asyncio
async def test_recompute_warns_and_picks_a_fresh_deterministic_split(world: World) -> None:
    build(world)
    request = world.request(['cars-a', 'cars-b'], MAPPING, holdout='recompute')
    preview, _ = await service.preview(world.fake, request)
    assert 'holdout_recompute_contamination' in [w.code for w in preview.warnings]
    await run_job(world, request)
    items = world.items('combined')
    by_class: dict[int, list[str]] = {}
    for doc in items.values():
        by_class.setdefault(doc['class_id'], []).append(doc['crop_id'])
    chosen, _ = select_test_holdout(by_class)
    assert {d['crop_id'] for d in items.values() if d['test_holdout']} == set(chosen)


@pytest.mark.asyncio
async def test_none_clears_every_holdout_flag(world: World) -> None:
    build(world)
    await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING, holdout='none'))
    assert not holdout_images(world)

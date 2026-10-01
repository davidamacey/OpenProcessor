"""A duplicate image's region boxes are merged into the priority copy, never
dropped, and never rewrite a region set a human or an import validated."""

from __future__ import annotations

import pytest

from src.config.region_fields import get_region_fields
from src.services.curation.region_boxes import read_boxes

from .world import World, run_job


F = get_region_fields()
CAR = [0.1, 0.1, 0.9, 0.9]
WHEEL = [0.15, 0.6, 0.3, 0.8]
OTHER_WHEEL = [0.6, 0.6, 0.8, 0.8]
SEED = 900
MAPPING = {
    'a': [
        {'dataset_class': 'car', 'action': 'create', 'new_class_name': 'car'},
        {'dataset_class': 'wheel', 'action': 'region'},
    ],
    'b': [
        {'dataset_class': 'car', 'action': 'map', 'new_class_name': 'car'},
        {'dataset_class': 'wheel', 'action': 'region'},
    ],
}


def _world(world: World) -> None:
    world.project('a', ['car', 'wheel'])
    world.project('b', ['car', 'wheel'])


def _car(world: World) -> dict:
    return next(d for d in world.items('combined').values() if d.get('class_name') == 'car')


def _boxes(world: World) -> list[tuple[float, ...]]:
    return sorted(tuple(b.bbox_norm) for b in read_boxes(_car(world), F))


@pytest.mark.asyncio
async def test_a_validated_region_box_of_a_duplicate_image_is_kept(world: World) -> None:
    _world(world)
    world.add_image('a', seed=SEED, items=[{'cls': 'car', 'bbox': CAR}])
    world.add_image(
        'b',
        seed=SEED,
        items=[
            {'cls': 'car', 'bbox': CAR},
            {'cls': 'wheel', 'bbox': WHEEL, 'class_validated': True},
        ],
    )
    store, _ = await run_job(world, world.request(['a', 'b'], MAPPING))
    assert _boxes(world) == [tuple(WHEEL)]
    assert store.job.read()['report']['regions_attached'] == 1


@pytest.mark.asyncio
async def test_a_box_the_priority_copy_already_holds_is_not_added_twice(world: World) -> None:
    _world(world)
    for slug in ('a', 'b'):
        world.add_image(
            slug,
            seed=SEED,
            items=[{'cls': 'car', 'bbox': CAR}, {'cls': 'wheel', 'bbox': WHEEL}],
        )
    store, _ = await run_job(world, world.request(['a', 'b'], MAPPING))
    assert _boxes(world) == [tuple(WHEEL)]
    assert store.job.read()['report']['regions_already_present'] == 1


@pytest.mark.asyncio
async def test_a_validated_region_set_is_not_rewritten_by_a_duplicates_box(world: World) -> None:
    _world(world)
    locked = {
        F.boxes: [
            {
                'box_id': 'b1',
                'bbox_norm': WHEEL,
                'state': 'accepted',
                'source': 'human',
                'detector': None,
            }
        ],
        F.validated: True,
        F.label_source: 'human',
    }
    world.add_image('a', seed=SEED, items=[{'cls': 'car', 'bbox': CAR, **locked}])
    world.add_image(
        'b',
        seed=SEED,
        items=[{'cls': 'car', 'bbox': CAR}, {'cls': 'wheel', 'bbox': OTHER_WHEEL}],
    )
    await run_job(world, world.request(['a', 'b'], MAPPING))
    car = _car(world)
    assert [b.bbox_norm for b in read_boxes(car, F)] == [tuple(WHEEL)]
    assert car[F.validated] is True
    assert car[F.label_source] == 'human'
    standalone = [d for d in world.items('combined').values() if d.get('import_standalone_region')]
    assert [d['bbox_norm'] for d in standalone] == [OTHER_WHEEL]

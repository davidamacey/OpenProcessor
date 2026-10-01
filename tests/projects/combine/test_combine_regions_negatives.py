"""Region-class mappings, reviewed negatives and empty frames across a
combine (W10 semantics, project source)."""

from __future__ import annotations

import pytest

from src.config.region_fields import get_region_fields
from src.services.curation.region_boxes import read_boxes

from .world import World, run_job


CAR = [0.1, 0.1, 0.9, 0.9]
WHEEL_IN = [0.15, 0.6, 0.3, 0.8]
WHEEL_OUT = [0.0, 0.0, 0.05, 0.05]  # inside no car
MAPPING = {
    'p': [
        {'dataset_class': 'car', 'action': 'create', 'new_class_name': 'car'},
        {'dataset_class': 'wheel', 'action': 'region'},
        {'dataset_class': 'bus', 'action': 'create', 'new_class_name': 'bus'},
    ]
}


@pytest.mark.asyncio
async def test_a_region_class_becomes_a_region_box_on_its_parent(world: World) -> None:
    world.project('p', ['car', 'wheel', 'bus'])
    world.add_image(
        'p',
        items=[
            {'cls': 'car', 'bbox': CAR},
            {'cls': 'wheel', 'bbox': WHEEL_IN, 'class_validated': True},
            {'cls': 'wheel', 'bbox': WHEEL_OUT},
        ],
    )
    store, _ = await run_job(world, world.request(['p'], MAPPING))
    F = get_region_fields()
    items = list(world.items('combined').values())
    parent = next(d for d in items if d.get('class_name') == 'car')
    boxes = read_boxes(parent, F)
    assert [(b.bbox_norm, b.state) for b in boxes] == [(tuple(WHEEL_IN), 'accepted')]
    assert parent[F.count] == 1
    standalone = [d for d in items if d.get('import_standalone_region')]
    assert len(standalone) == 1  # the wheel inside no car
    assert standalone[0]['bbox_norm'] == WHEEL_OUT
    assert 'class_name' not in standalone[0]
    # The wheel is never an item of its own class.
    assert {d.get('class_name') for d in items} == {'car', None}
    report = store.job.read()['report']
    assert (report['regions_attached'], report['regions_standalone']) == (1, 1)


@pytest.mark.asyncio
async def test_reviewed_negatives_carry_their_mapped_class_names(world: World) -> None:
    world.project('p', ['car', 'bus'])
    neg, _ = world.add_image('p', negative=True)
    world.add_image('p', items=[{'cls': 'car', 'bbox': CAR}])
    mapping = {
        'p': [
            {'dataset_class': 'car', 'action': 'create', 'new_class_name': 'vehicle'},
            {'dataset_class': 'bus', 'action': 'skip'},
        ]
    }
    # `bus` has no item with a label, so it needs no row; `car` is renamed.
    await run_job(world, world.request(['p'], {'p': [mapping['p'][0]]}))
    frames = [d for d in world.images('combined').values() if d.get('import_label_state')]
    (frame,) = frames
    assert frame['origin_image_id'] == neg
    # The negative said "no car, no bus": only classes that exist in the target
    # carry over, under their TARGET names.
    assert frame['negative_for'] == ['vehicle']


@pytest.mark.asyncio
async def test_a_frame_with_no_items_is_copied_as_a_frame(world: World) -> None:
    world.project('p', ['car'])
    empty, _ = world.add_image('p')
    world.add_image('p', items=[{'cls': 'car', 'bbox': CAR}])
    await run_job(world, world.request(['p'], {'p': MAPPING['p'][:1]}))
    origins = {d['origin_image_id'] for d in world.images('combined').values()}
    assert empty in origins
    assert len(world.items('combined')) == 1

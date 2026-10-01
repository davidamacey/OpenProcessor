"""Items a combine cannot map or carry whole: no class, a vector of the
wrong size."""

from __future__ import annotations

import pytest

from .test_combine_execute import MAPPING, S1, S2
from .world import DIM, World, run_job


@pytest.mark.asyncio
async def test_an_item_with_no_class_is_copied_unclassed(world: World) -> None:
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['sedan', 'lorry'])
    world.add_image('cars-a', items=[{'cls': 'car', 'bbox': S1}, {'bbox': S2}])
    store, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    by_box = {tuple(d['bbox_norm']): d for d in world.items('combined').values()}
    assert by_box[tuple(S1)]['class_name'] == 'car'
    unclassed = by_box[tuple(S2)]
    assert 'class_id' not in unclassed
    assert 'class_name' not in unclassed
    assert store.job.read()['report']['items_copied'] == 2


@pytest.mark.asyncio
async def test_an_embedding_of_the_wrong_size_is_dropped_and_counted(world: World) -> None:
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['sedan', 'lorry'])
    world.add_image(
        'cars-a',
        items=[
            {'cls': 'car', 'bbox': S1, 'pe_embedding': [0.3] * (DIM + 1)},
            {'cls': 'truck', 'bbox': S2, 'pe_embedding': [0.3] * DIM},
        ],
    )
    store, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    by_class = {d['class_name']: d for d in world.items('combined').values()}
    assert 'pe_embedding' not in by_class['car']
    assert by_class['truck']['pe_embedding'] == [0.3] * DIM
    assert store.job.read()['report']['embeddings_dropped'] == 1

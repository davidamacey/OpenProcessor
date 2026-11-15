"""A combine's images doc must not carry the source's item list or writer id: the target's items
have ids of their own, so a copied list would make every combined image look incomplete."""

from __future__ import annotations

import pytest

from .test_combine_execute import MAPPING
from .world import World, run_job


@pytest.mark.asyncio
async def test_a_combined_image_carries_no_item_list_no_mark_and_no_source_write_id(
    world: World,
) -> None:
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['sedan', 'lorry'])
    image_id, _ = world.add_image('cars-a', items=[{'cls': 'car', 'bbox': [0.1, 0.1, 0.5, 0.5]}])
    world.images('cars-a')[image_id].update(
        item_ids=['the-source-item'], ingest_incomplete=True, ingest_write_id='source-writer'
    )

    run, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))

    assert run.status == 'completed', run.error
    [image] = world.images('combined').values()
    assert not {'item_ids', 'ingest_incomplete', 'ingest_write_id'} & set(image)

"""A combine never writes an item document at the nested-document limit (OpenSearch k-NN issue
3623): the item is not copied, it is counted and its id is in the report. Nothing is truncated."""

from __future__ import annotations

from typing import Any

import pytest

from .test_combine_execute import MAPPING, S1, S2
from .world import DIM, World, run_job


def _boxes(n: int) -> list[dict[str, Any]]:
    return [{'box_id': f'b{i}', 'bbox_norm': list(S1), 'state': 'accepted'} for i in range(n)]


def _entries(n: int) -> list[dict[str, Any]]:
    return [{'box_id': f'b{i}', 'bbox_norm': list(S1), 'embedding': [0.3] * DIM} for i in range(n)]


@pytest.mark.asyncio
async def test_an_item_over_the_limit_is_not_copied_and_is_reported_with_its_id(
    world: World,
) -> None:
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['sedan', 'lorry'])
    world.add_image(
        'cars-a',
        items=[
            {
                'cls': 'car',
                'bbox': S1,
                'region_boxes': _boxes(75),
                'region_box_embeddings': _entries(75),
            },
            {
                'cls': 'truck',
                'bbox': S2,
                'region_boxes': _boxes(74),
                'region_box_embeddings': _entries(0),
            },
        ],
    )
    run, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    copied = {d['class_name']: d for d in world.items('combined').values()}
    assert set(copied) == {'truck'}
    assert len(copied['truck']['region_boxes']) == 74  # untouched, never truncated
    assert run.report['items_over_nested_limit'] == 1
    [(source_item_id,)] = [(i,) for i in run.report['over_nested_limit_ids']]
    assert source_item_id.startswith('cars-a:')
    assert run.report['items_copied'] == 1


@pytest.mark.asyncio
async def test_an_item_just_below_the_limit_is_copied_whole(world: World) -> None:
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['sedan', 'lorry'])
    world.add_image(
        'cars-a',
        items=[
            {
                'cls': 'car',
                'bbox': S1,
                'region_boxes': _boxes(74),
                'region_box_embeddings': _entries(74),
            }
        ],
    )
    run, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    [copied] = world.items('combined').values()
    assert len(copied['region_boxes']) == 74
    assert len(copied['region_box_embeddings']) == 74
    assert run.report.get('items_over_nested_limit', 0) == 0
    assert 'over_nested_limit_ids' not in run.report

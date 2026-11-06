"""A combine carries a vector only into a space the target has (research embeddings 5.4).

Vectors belong to a space (``space_key``: model and version). A source whose primary space has
a key the target lacks cannot have its vectors compared with the target's, whatever their
dimension: the vectors are dropped, the items are marked ``deferred`` for re-embedding, and
the preview warns by space key."""

from __future__ import annotations

import pytest

from .test_combine_execute import MAPPING, S1, S2
from .world import DEFAULT_KEY, DIM, World, run_job


OTHER = 'mobileclip2_s2_v1'


def _sources(world: World, *, other: str | None = None) -> None:
    world.project('cars-a', ['car', 'truck'])
    world.project('cars-b', ['sedan', 'lorry'])
    if other is not None:
        world.space_keys['cars-a'] = other
    world.add_image(
        'cars-a',
        items=[
            {'cls': 'car', 'bbox': S1, 'pe_embedding': [0.3] * DIM, 'embedding_state': 'embedded'}
        ],
        vector=True,
    )


@pytest.mark.asyncio
async def test_vectors_of_a_space_the_target_has_are_copied(world: World) -> None:
    _sources(world)
    run, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    [item] = world.items('combined').values()
    [image] = world.images('combined').values()
    assert item['pe_embedding'] == [0.3] * DIM
    assert item['embedding_state'] == 'embedded'
    assert image['pe_embedding'] == [0.1] * DIM
    assert run.report.get('embeddings_dropped', 0) == 0


@pytest.mark.asyncio
async def test_vectors_of_another_space_are_dropped_even_at_the_same_dimension(
    world: World,
) -> None:
    _sources(world, other=OTHER)
    assert OTHER != DEFAULT_KEY
    run, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    [item] = world.items('combined').values()
    [image] = world.images('combined').values()
    assert 'pe_embedding' not in item
    assert item['embedding_state'] == 'deferred'
    assert 'pe_embedding' not in image
    assert run.report['embeddings_dropped'] == 1


@pytest.mark.asyncio
async def test_each_source_is_judged_by_its_own_space(world: World) -> None:
    _sources(world)
    world.space_keys['cars-b'] = OTHER
    world.add_image(
        'cars-b',
        items=[
            {
                'cls': 'sedan',
                'bbox': S2,
                'pe_embedding': [0.4] * DIM,
                'embedding_state': 'embedded',
            }
        ],
    )
    run, _ = await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    by_origin = {d['origin_project']: d for d in world.items('combined').values()}
    assert by_origin['cars-a']['pe_embedding'] == [0.3] * DIM
    assert 'pe_embedding' not in by_origin['cars-b']
    assert by_origin['cars-b']['embedding_state'] == 'deferred'
    assert run.report['embeddings_dropped'] == 1

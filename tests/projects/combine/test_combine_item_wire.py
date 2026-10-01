"""A combined item's provenance reaches the wire: ``origin_*`` says where the
copy came from and ``combine_*`` carries the conflict a human must decide.

Uses the real combine job (not a hand-built doc), so the wire is checked
against what the combiner actually stores.
"""

from __future__ import annotations

import pytest

from src.routers.curation._item_models import ItemDoc
from src.services.curation.wire import ITEM_WIRE_KEYS, serialize_item

from .test_combine_execute import MAPPING, build, target_item
from .world import World, run_job


COMBINE_KEYS = {
    'origin_project',
    'origin_item_id',
    'origin_image_id',
    'origin_split',
    'combine_conflict',
    'combine_conflict_origins',
    'combine_merged_origins',
}


def test_the_keys_are_on_the_wire_contract_and_the_model() -> None:
    assert COMBINE_KEYS <= ITEM_WIRE_KEYS
    assert set(ItemDoc.model_fields) >= COMBINE_KEYS


def test_an_item_that_never_went_through_a_combine_serves_neutral_values() -> None:
    wire = serialize_item({'crop_id': 'c1'})
    assert {k: wire[k] for k in COMBINE_KEYS} == {
        'origin_project': None,
        'origin_item_id': None,
        'origin_image_id': None,
        'origin_split': None,
        'combine_conflict': False,
        'combine_conflict_origins': [],
        'combine_merged_origins': [],
    }


@pytest.mark.asyncio
async def test_combined_items_serve_their_origin_and_conflict(world: World) -> None:
    sc = build(world)
    await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))

    conflict = target_item(world, sc, sc.crops['a_s2'], 'cars-a')
    wire = serialize_item(conflict)
    assert wire['origin_project'] == 'cars-a'
    assert wire['origin_item_id'] == sc.crops['a_s2']
    assert wire['origin_image_id'] == sc.a_shared
    assert wire['combine_conflict'] is True
    assert wire['combine_conflict_origins'] == sorted(
        [f'cars-a:{sc.crops["a_s2"]}', f'cars-b:{sc.crops["b_s2"]}']
    )
    assert ItemDoc.model_validate(wire).combine_conflict is True

    merged = serialize_item(target_item(world, sc, sc.crops['b_s1'], 'cars-b'))
    assert merged['combine_conflict'] is False
    assert merged['combine_merged_origins'] == sorted(
        [f'cars-a:{sc.crops["a_s1"]}', f'cars-b:{sc.crops["b_s1"]}']
    )

    plain = serialize_item(target_item(world, sc, sc.crops['a2_truck'], 'cars-a'))
    assert plain['origin_project'] == 'cars-a'
    assert plain['combine_conflict'] is False
    assert plain['combine_conflict_origins'] == []

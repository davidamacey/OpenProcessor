"""The export collapses near-duplicate frames from the stored groups of the current run."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.domain.near_dup import Seat, collapse_drops
from src.services.near_duplicates import collapse
from src.services.near_duplicates.build import full_build

from .engine import IMAGES, image_docs, make_env


@dataclass
class Row:
    image_id: str
    has_test_crop: bool = False
    item: str = ''


def _seat(group: str, rep: bool, cosine: float) -> Seat:
    return Seat(group, rep, cosine)


def test_one_survivor_per_group_the_representative_unless_a_frame_is_held_out() -> None:
    seats = {
        'a': _seat('g', True, 1.0),
        'b': _seat('g', False, 0.99),
        'c': _seat('g', False, 0.98),
        'x': _seat('h', True, 1.0),
        'y': _seat('h', False, 0.99),
    }
    assert collapse_drops(seats) == {'b', 'c', 'y'}
    assert collapse_drops(seats, preferred={'c', 'y'}) == {'a', 'b', 'x'}


def test_without_the_representative_the_closest_member_survives_and_ties_go_to_the_smaller_id() -> (
    None
):
    seats = {
        'b': _seat('g', False, 0.99),
        'c': _seat('g', False, 0.99),
        'd': _seat('g', False, 0.97),
    }
    assert collapse_drops(seats) == {'c', 'd'}


def test_a_frame_in_no_group_is_never_dropped() -> None:
    assert collapse_drops({'a': _seat('g', True, 1.0)}) == set()


@pytest.mark.db
@pytest.mark.asyncio
async def test_collapse_reads_the_current_run_and_keeps_every_row_of_a_survivor(
    pg, project, monkeypatch
) -> None:
    docs = {IMAGES: image_docs({'burst': ['a1', 'a2', 'a3'], 'pair': ['b1', 'b2']}, ['s1'])}
    env, _ = await make_env(pg, project, docs)
    await full_build(env)

    async def bound() -> object:
        return project

    monkeypatch.setattr(collapse, 'bound_project_id', bound)
    rows = [
        Row('a1', item='i1'),
        Row('a1', item='i2'),
        Row('a2', item='i3'),
        Row('a3', item='i4'),
        Row('b1', item='i5'),
        Row('b2', has_test_crop=True, item='i6'),
        Row('s1', item='i7'),
        Row('', item='i8'),
    ]

    kept, stats = await collapse.collapse_near_duplicates(rows)

    survivors = {r.image_id for r in kept}
    assert len(survivors & {'a1', 'a2', 'a3'}) == 1
    assert 'b2' in survivors
    assert 'b1' not in survivors
    assert {'s1', ''} <= survivors
    survivor = (survivors & {'a1', 'a2', 'a3'}).pop()
    assert sorted(r.item for r in kept if r.image_id == survivor) == sorted(
        r.item for r in rows if r.image_id == survivor
    )
    assert (stats['frames_dropped'], stats['near_dup_groups'], stats['n_output_rows']) == (
        3,
        2,
        len(kept),
    )


@pytest.mark.db
@pytest.mark.asyncio
async def test_collapse_refuses_when_the_project_has_no_run(pg, project, monkeypatch) -> None:
    async def bound() -> object:
        return project

    monkeypatch.setattr(collapse, 'bound_project_id', bound)
    with pytest.raises(collapse.NoCurrentRunError):
        await collapse.collapse_near_duplicates([Row('a1')])

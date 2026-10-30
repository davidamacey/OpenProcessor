"""The loop is reproducible: the same members, seed and path split the same way whatever
order the ids arrive in, and two runs with the same seed and the same VLM write the same
event log; a different seed draws a different sample.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from curation.assist_loop_fakes import FakeVlm, World, drive, make_world, without_times
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation.assist_loop.splitter import SplitResult, fetch_embeddings, split_node


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.assist_loop.store import LoopStore


pytestmark = pytest.mark.asyncio

ITEMS = base_curation_config().items_index
LOOP = 'alp_20261028T000000Z_000002'


def _blobs(per_blob: int = 12, centres: int = 4, dim: int = 8) -> dict[str, list[float]]:
    rng = np.random.default_rng(3)
    out: dict[str, list[float]] = {}
    for c in range(centres):
        base = np.zeros(dim)
        base[c] = 1.0
        for i in range(per_blob):
            out[f'x{c}-{i:03d}'] = (base + rng.normal(0, 0.02, dim)).tolist()
    return out


def _shuffled(vectors: dict[str, list[float]], seed: int) -> dict[str, list[float]]:
    keys = list(vectors)
    random.Random(seed).shuffle(keys)
    return {k: vectors[k] for k in keys}


async def _split(vectors: dict[str, list[float]], **kw: object) -> SplitResult:
    options: dict[str, object] = {'branching': 4, 'seed': 7, 'path': 'r10001'}
    return await split_node(list(vectors), vectors, **{**options, **kw})  # type: ignore[arg-type]


async def test_children_follow_the_structure_and_partition_the_members() -> None:
    vectors = _blobs()

    result = await _split(vectors)

    assert len(result.children) == 4
    assert sorted(i for child in result.children for i in child) == sorted(vectors)
    for child in result.children:
        assert len({i.split('-')[0] for i in child}) == 1, 'a child mixes blobs'
        assert child == sorted(child)
    assert result.unplaced == []


async def test_children_are_ordered_by_their_smallest_id() -> None:
    result = await _split(_blobs())

    firsts = [child[0] for child in result.children]
    assert firsts == sorted(firsts)


async def test_input_order_does_not_change_the_split() -> None:
    vectors = _blobs()

    base = await _split(vectors)
    for seed in (1, 2, 3):
        assert (await _split(_shuffled(vectors, seed))).children == base.children


async def test_above_the_refine_cap_a_seeded_kmeans_splits_instead() -> None:
    vectors = _blobs(per_blob=15)

    first = await _split(vectors, max_members=20)
    again = await _split(_shuffled(vectors, 5), max_members=20)

    assert first.method == 'kmeans'
    assert len(first.children) == 4
    assert first.children == again.children
    for child in first.children:
        assert len({i.split('-')[0] for i in child}) == 1


async def test_within_the_cap_the_refine_core_is_used() -> None:
    assert (await _split(_blobs())).method == 'ahc'


async def test_a_node_too_small_to_split_has_no_children() -> None:
    vectors = dict(list(_blobs().items())[:3])

    result = await _split(vectors)

    assert result.children == []


async def test_a_split_that_does_not_separate_anything_is_not_a_split() -> None:
    same = {f'x{i:03d}': [1.0, 0.0, 0.0, 0.0] for i in range(30)}

    result = await _split(same, max_members=10)

    assert result.children == []


async def test_members_without_an_embedding_are_unplaced_not_guessed() -> None:
    vectors = _blobs()
    ids = [*vectors, 'ghost']

    result = await split_node(ids, vectors, branching=4, seed=7, path='r10001')

    assert result.unplaced == ['ghost']
    assert sorted(i for child in result.children for i in child) == sorted(vectors)


async def test_fetch_embeddings_reads_the_residual_field() -> None:
    from src.services.curation.clustering.residual_fetch import RESIDUAL_EMBEDDING_FIELD

    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'a': {'crop_id': 'a', RESIDUAL_EMBEDDING_FIELD: [1.0, 0.0]},
                'b': {'crop_id': 'b', RESIDUAL_EMBEDDING_FIELD: [0.0, 1.0]},
                'c': {'crop_id': 'c'},
            }
        }
    )

    got = await fetch_embeddings(fake, ['a', 'b', 'c', 'missing'])  # type: ignore[arg-type]

    assert got == {'a': [1.0, 0.0], 'b': [0.0, 1.0]}


# ---------------------------------------------------------------- the whole loop


def _loop_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    world = make_world(tmp_path, monkeypatch)
    world.add_cluster(10001, {'car': 60, 'truck': 60, 'bus': 60, 'van': 60}, prefix='a')
    world.add_cluster(10002, {'car': 150}, prefix='b')
    return world


def _events(store: LoopStore) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in without_times(e).items() if k not in ('seq',)} for e in store.events()
    ]


def _by_node(store: LoopStore) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for event in _events(store):
        event.pop('rid', None)
        out.setdefault(event.get('path', ''), []).append(event)
    return out


def _sampled(store: LoopStore) -> set[str]:
    return {
        crop_id
        for e in store.events()
        if e['event'] == 'answers'
        for crop_id in [*e['answers'], *e['failed']]
    }


@pytest.mark.usefixtures('readable_crops')
async def test_the_same_seed_gives_the_same_event_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _loop_world(tmp_path, monkeypatch)
    pristine = world.snapshot()
    serial = {'node_concurrency': 1, 'vlm_concurrency': 1}

    first = await drive(world, tmp_path / 'one' / LOOP, FakeVlm(world.truth), **serial)
    world.restore(pristine)
    second = await drive(world, tmp_path / 'two' / LOOP, FakeVlm(world.truth), **serial)

    assert _events(first.store) == _events(second.store)


@pytest.mark.usefixtures('readable_crops')
async def test_the_same_seed_gives_the_same_node_events_under_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _loop_world(tmp_path, monkeypatch)
    pristine = world.snapshot()

    first = await drive(world, tmp_path / 'one' / LOOP, FakeVlm(world.truth))
    world.restore(pristine)
    second = await drive(world, tmp_path / 'two' / LOOP, FakeVlm(world.truth))

    assert _by_node(first.store) == _by_node(second.store)


@pytest.mark.usefixtures('readable_crops')
async def test_a_different_seed_draws_a_different_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _loop_world(tmp_path, monkeypatch)
    pristine = world.snapshot()

    first = await drive(world, tmp_path / 'one' / LOOP, FakeVlm(world.truth), seed=0)
    world.restore(pristine)
    second = await drive(world, tmp_path / 'two' / LOOP, FakeVlm(world.truth), seed=1)

    assert _sampled(first.store) != _sampled(second.store)

"""The loop's partition tree is reproducible: the same members, seed and path split the same
way whatever order the ids arrive in (the split part; the run-level cases come with the driver).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation.assist_loop.splitter import SplitResult, fetch_embeddings, split_node


pytestmark = pytest.mark.asyncio

ITEMS = base_curation_config().items_index


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

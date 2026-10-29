"""``refine_members(n_clusters=...)``: split a member set into exactly k groups.

The auto-label loop splits a node that failed its gate into a fixed number of children,
through the same AHC core the refine route uses; ``distance_threshold`` stays the default.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.services.curation.clustering.refine import refine_members


pytestmark = pytest.mark.asyncio


def _blobs(per_blob: int, centres: int = 4, dim: int = 8, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    members: list[dict[str, Any]] = []
    for c in range(centres):
        base = np.zeros(dim)
        base[c] = 1.0
        for i in range(per_blob):
            vec = base + rng.normal(0, 0.02, dim)
            members.append({'_id': f'b{c}-{i:03d}', 'embedding': vec.tolist(), 'class_name': None})
    return members


async def _run(members: list[dict[str, Any]], **kw: Any) -> tuple[dict[str, Any], dict[str, str]]:
    written: dict[str, str] = {}

    async def count() -> int:
        return len(members)

    async def fetch() -> list[dict[str, Any]]:
        return members

    async def write(pairs: list[tuple[Any, str]]) -> int:
        written.update(dict(pairs))
        return len(pairs)

    summary = await refine_members(
        7, count_members=count, fetch_members=fetch, write_subids=write, unit='items', **kw
    )
    return summary, written


async def test_n_clusters_gives_exactly_that_many_groups_following_the_structure() -> None:
    summary, written = await _run(_blobs(10), n_clusters=4)

    assert summary['action'] == 'refined'
    assert summary['n_subclusters'] == 4
    assert summary['distance_threshold'] is None
    by_blob = {}
    for crop_id, subid in written.items():
        by_blob.setdefault(crop_id.split('-')[0], set()).add(subid)
    assert all(len(subids) == 1 for subids in by_blob.values()), 'a blob was split'
    assert len({next(iter(s)) for s in by_blob.values()}) == 4


async def test_n_clusters_overrides_the_distance_threshold() -> None:
    members = _blobs(10)
    with_threshold, _ = await _run(members, distance_threshold=0.0001)
    fixed, _ = await _run(members, distance_threshold=0.0001, n_clusters=2)

    assert with_threshold['n_subclusters'] > 4
    assert fixed['n_subclusters'] == 2


async def test_fewer_members_than_clusters_is_skipped_not_an_error() -> None:
    summary, written = await _run(_blobs(1, centres=5)[:5], n_clusters=6)

    assert summary['action'] == 'skipped_too_small'
    assert written == {}


async def test_the_default_still_cuts_at_the_distance_threshold() -> None:
    summary, _ = await _run(_blobs(10))

    assert summary['n_subclusters'] == 4
    assert summary['distance_threshold'] == pytest.approx(0.25)


async def test_a_single_group_is_not_a_split() -> None:
    with pytest.raises(ValueError, match='n_clusters'):
        await _run(_blobs(10), n_clusters=1)

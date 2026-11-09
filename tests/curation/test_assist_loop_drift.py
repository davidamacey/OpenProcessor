"""Clusters moving under the loop: a node whose members drifted out of its root by more than
10 % turns ``stale`` and writes nothing; below that, the moved members are skipped. The
loop's own sampled items are not drift: a class write leaves an item in its cluster.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, decisions, drive, make_world


if TYPE_CHECKING:
    from collections.abc import Sequence

    from curation.assist_loop_fakes import LoopEnv
    from src.services.curation.assist_loop.partition import (
        PartitionReader,
        PartitionScope,
        RootRef,
        RootSize,
    )

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('readable_crops')]


class RetrainingReader:
    """Just before a node's drift check, a re-projection moves ``share`` of the candidates of
    the named roots (by display number) to a cluster the pinned partition does not have."""

    def __init__(self, real: PartitionReader, world: World, shares: dict[int, float]) -> None:
        self.real = real
        self.world = world
        self.shares = shares
        self.moved: set[str] = set()

    async def pin(self) -> uuid.UUID:
        return await self.real.pin()

    async def survey(self, scope: PartitionScope) -> list[RootSize]:
        return await self.real.survey(scope)

    async def snapshots(self, scope: PartitionScope) -> list[RootRef]:
        return await self.real.snapshots(scope)

    async def root_of(self, ids: Sequence[str]) -> dict[str, str]:
        docs = self.world.os.docs(ITEMS)
        by_root: dict[int, list[str]] = {}
        for crop_id in sorted(ids):
            by_root.setdefault(self.world.number_of[crop_id], []).append(crop_id)
        for root, members in by_root.items():
            share = self.shares.get(root, 0.0)
            for crop_id in members[: int(len(members) * share)]:
                docs[crop_id]['cluster_uid'] = str(uuid.uuid4())
                self.moved.add(crop_id)
        return await self.real.root_of(ids)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(monkeypatch)
    w.add_cluster(10001, {'car': 200}, prefix='a')
    w.add_cluster(10002, {'truck': 200}, prefix='b')
    return w


def _run(world: World, loop_env: LoopEnv, shares: dict[int, float]) -> tuple[Any, RetrainingReader]:
    wrapped: list[RetrainingReader] = []

    def wrap(real: PartitionReader) -> RetrainingReader:
        wrapped.append(RetrainingReader(real, world, shares))
        return wrapped[0]

    run = asyncio.run(drive(world, loop_env, FakeVlm(world.truth), wrap_reader=wrap, max_depth=0))
    return run, wrapped[0]


def test_a_node_over_the_drift_limit_turns_stale_and_writes_nothing(
    world: World, loop_env: LoopEnv
) -> None:
    run, _ = _run(world, loop_env, {10001: 0.2})

    assert decisions(run.record)['r10001'] == 'stale'
    for crop_id, doc in world.os.docs(ITEMS).items():
        if crop_id.startswith('a-'):
            assert doc['class_source'] != 'assist_propagated', crop_id


def test_below_the_limit_the_node_passes_and_moved_members_are_never_written(
    world: World, loop_env: LoopEnv
) -> None:
    run, reader = _run(world, loop_env, {10002: 0.05})

    assert decisions(run.record)['r10002'] == 'pass'
    assert reader.moved
    for crop_id in reader.moved:
        assert world.doc(crop_id)['class_source'] == 'item_proposal', crop_id
    written = [
        c
        for c, d in world.os.docs(ITEMS).items()
        if c.startswith('b-') and d['class_source'] == 'assist_propagated'
    ]
    assert len(written) > 100


def test_the_loops_own_sampled_items_are_not_drift(world: World, loop_env: LoopEnv) -> None:
    run, _ = _run(world, loop_env, {})

    assert decisions(run.record) == {'r10001': 'pass', 'r10002': 'pass'}
    sampled = [c for c, d in world.os.docs(ITEMS).items() if d.get('class_source') == 'vlm']
    assert sampled, 'the VLM labeled a sample'
    assert all(
        world.doc(c)['cluster_uid'] == str(world.uids[10001]) for c in sampled if c.startswith('a-')
    )

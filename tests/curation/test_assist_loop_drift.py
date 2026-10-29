"""Clusters moving under the loop: a node whose members drifted out of its root by more than
10 % turns ``stale`` and writes nothing; below that, the moved members are skipped. The
loop's own sampled items moving (the normalizer placing VLM-labeled items in their class
cluster) is not drift.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, decisions, drive, make_world
from src.services.curation.assist_loop.partition import ClusterFieldPartition


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('readable_crops')

MOVED_TO = 99999


class RetrainingReader(ClusterFieldPartition):
    """Just before a node's drift check, a retrain moves ``share`` of the candidates of
    the named roots elsewhere, and the normalizer moves every VLM-labeled item into its
    class cluster."""

    def __init__(self, world: World, shares: dict[int, float]) -> None:
        super().__init__(world.os)  # type: ignore[arg-type]
        self.world = world
        self.shares = shares
        self.moved: set[str] = set()

    async def root_of(self, ids: Sequence[str]) -> dict[str, str]:
        docs = self.world.os.docs(ITEMS)
        for doc in docs.values():
            if doc.get('class_source') == 'vlm':
                doc['cluster_id'] = doc['class_id']
        by_root: dict[int, list[str]] = {}
        for crop_id in sorted(ids):
            by_root.setdefault(docs[crop_id]['cluster_id'], []).append(crop_id)
        for root, members in by_root.items():
            share = self.shares.get(root, 0.0)
            for crop_id in members[: int(len(members) * share)]:
                docs[crop_id]['cluster_id'] = MOVED_TO
                self.moved.add(crop_id)
        return await super().root_of(ids)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(tmp_path, monkeypatch)
    w.add_cluster(10001, {'car': 200}, prefix='a')
    w.add_cluster(10002, {'truck': 200}, prefix='b')
    return w


def _run(world: World, tmp_path: Path, reader: RetrainingReader) -> Any:
    return asyncio.run(
        drive(world, tmp_path / 'loop', FakeVlm(world.truth), reader=reader, max_depth=0)
    )


def test_a_node_over_the_drift_limit_turns_stale_and_writes_nothing(
    world: World, tmp_path: Path
) -> None:
    reader = RetrainingReader(world, {10001: 0.2})

    run = _run(world, tmp_path, reader)

    assert decisions(run.store)['r10001'] == 'stale'
    for crop_id, doc in world.os.docs(ITEMS).items():
        if crop_id.startswith('a-'):
            assert doc['class_source'] != 'assist_propagated', crop_id


def test_below_the_limit_the_node_passes_and_moved_members_are_never_written(
    world: World, tmp_path: Path
) -> None:
    reader = RetrainingReader(world, {10002: 0.05})

    run = _run(world, tmp_path, reader)

    assert decisions(run.store)['r10002'] == 'pass'
    assert reader.moved
    for crop_id in reader.moved:
        assert world.doc(crop_id)['class_source'] == 'item_proposal', crop_id
    written = [
        c
        for c, d in world.os.docs(ITEMS).items()
        if c.startswith('b-') and d['class_source'] == 'assist_propagated'
    ]
    assert len(written) > 100


def test_the_loops_own_sampled_items_moving_is_not_drift(world: World, tmp_path: Path) -> None:
    reader = RetrainingReader(world, {})

    run = _run(world, tmp_path, reader)

    assert decisions(run.store) == {'r10001': 'pass', 'r10002': 'pass'}
    sampled = [c for c, d in world.os.docs(ITEMS).items() if d.get('class_source') == 'vlm']
    assert sampled, 'the VLM-labeled sample was moved to its class cluster'
    assert all(world.doc(c)['cluster_id'] != 10001 for c in sampled if c.startswith('a-'))

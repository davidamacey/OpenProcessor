"""The loop's pinned partition: roots are the serving partition's clusters (named by uid, shown
by display number), members are the assigned items it serves, a snapshot per root is a blob in
the storage seam, and a reader never follows the project to another partition.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from curation.assist_loop_fakes import SLUG, World, make_world
from src.clients.storage.roots import RootedStore
from src.domain.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.assist.scope import membership_digest
from src.services.curation.assist_loop.partition import (
    NoServingPartitionError,
    PartitionReader,
    PartitionScope,
)
from src.services.curation.assist_loop.snapshots import SnapshotStore
from src.services.curation.class_write_guard import class_state_token


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from curation.assist_loop_fakes import LoopEnv
    from src.services.curation.assist_loop.partition import SessionFactory

pytestmark = [pytest.mark.db, pytest.mark.asyncio]

A, B, K = 10001, 10002, 4
ALL = PartitionScope(cluster_uids=None, min_root_size=1)


def _world(monkeypatch: pytest.MonkeyPatch) -> World:
    w = make_world(monkeypatch)
    w.add_cluster(A, {'car': 5}, prefix='a')
    human = {'class_id': 3, 'class_name': 'widget', 'class_validated': True}
    w.add_item('a-human', A, 'car', class_source='human', **human)
    w.add_item('a-valid', A, 'car', class_source='vlm', **human)
    w.add_item('a-frozen', A, 'car', test_holdout=True)
    w.add_item('a-shelved', A, 'car', class_excluded=True)
    w.add_item('a-review', A, 'car', cluster_state='review')
    w.add_cluster(B, {'truck': 3}, prefix='b')
    for i in range(6):
        w.add_item(f'k{i:02d}', K, 'bus', class_id=4, class_name='gadget', class_source='vlm')
    return w


@asynccontextmanager
async def _sessions(env: LoopEnv) -> AsyncIterator[SessionFactory]:
    engine = create_async_engine(env.pg.url('app'), poolclass=NullPool)
    try:
        yield await env.sessions_for(engine)
    finally:
        await engine.dispose()


@asynccontextmanager
async def _reader(world: World, env: LoopEnv) -> AsyncIterator[PartitionReader]:
    """A reader that has pinned the partition the world serves in ``env``'s project."""
    world.serve(env)
    async with _sessions(env) as sessions:
        reader = PartitionReader(
            world.os,  # type: ignore[arg-type]
            sessions,
            project=env.project,
        )
        assert await reader.pin() == world.served[env.project]
        yield reader


async def test_roots_are_the_serving_partitions_clusters_at_least_min_size(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    async with _reader(world, loop_env) as reader:
        survey = await reader.survey(PartitionScope(cluster_uids=None, min_root_size=7))

    assert [(s.key, s.cluster_uid) for s in survey] == [('r10001', str(world.uids[A]))]


async def test_roots_come_in_display_order(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    async with _reader(world, loop_env) as reader:
        survey = await reader.survey(ALL)

    assert [s.key for s in survey] == ['r4', 'r10001', 'r10002']


async def test_named_clusters_are_the_only_roots_and_an_unknown_uid_is_none(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    async with _reader(world, loop_env) as reader:
        survey = await reader.survey(
            PartitionScope(
                cluster_uids=[str(world.uids[B]), str(world.uids[K]), str(uuid.uuid4())],
                min_root_size=1,
            )
        )

    assert [s.key for s in survey] == ['r4', 'r10002']


async def test_a_root_snapshots_every_assigned_member_with_the_state_the_lock_rule_gives(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    async with _reader(world, loop_env) as reader:
        (root,) = await reader.snapshots(
            PartitionScope(cluster_uids=[str(world.uids[A])], min_root_size=1)
        )

    states = {m.crop_id: m.state for m in root.members}
    assert states == {
        'a-car-0000': 'eligible',
        'a-car-0001': 'eligible',
        'a-car-0002': 'eligible',
        'a-car-0003': 'eligible',
        'a-car-0004': 'eligible',
        'a-human': 'locked',
        'a-valid': 'locked',
        'a-frozen': 'holdout',
        'a-shelved': 'excluded',
    }, 'a review member is in no root: it never propagates'
    assert (root.key, root.display_no, root.cluster_uid) == ('r10001', A, str(world.uids[A]))
    assert [m.crop_id for m in root.members] == sorted(states)
    assert root.version == membership_digest(list(states))
    assert root.eligible_ids == [f'a-car-000{i}' for i in range(5)]
    human = next(m for m in root.members if m.crop_id == 'a-human')
    assert human.class_name == 'widget'
    for member in root.members:
        assert member.token == class_state_token(world.doc(member.crop_id))


async def test_only_a_person_or_a_validated_import_is_ground_truth(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    async with _reader(world, loop_env) as reader:
        (root,) = await reader.snapshots(
            PartitionScope(cluster_uids=[str(world.uids[A])], min_root_size=1)
        )

    truth = {m.crop_id for m in root.members if m.truth}

    assert truth == {'a-human'}, 'a class an automated writer validated is not evidence'


async def test_the_survey_counts_exactly_what_a_snapshot_would_make_eligible(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    world.add_item(
        'a-import-suggestion', A, 'car', class_source=LABEL_IMPORT_CLASS_SOURCE, class_name='widget'
    )
    world.add_item(
        'a-import-valid',
        A,
        'car',
        class_source=LABEL_IMPORT_CLASS_SOURCE,
        class_name='widget',
        class_validated=True,
    )
    async with _reader(world, loop_env) as reader:
        survey = await reader.survey(ALL)
        roots = await reader.snapshots(ALL)

    assert [(s.key, s.members, s.eligible) for s in survey] == [
        (r.key, len(r.members), len(r.eligible_ids)) for r in roots
    ]
    assert [(s.key, s.eligible, s.members) for s in survey] == [
        ('r4', 6, 6),
        ('r10001', 6, 11),
        ('r10002', 3, 3),
    ]


async def test_root_of_names_the_cluster_each_item_is_in_now(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    world.doc('b-truck-0002')['cluster_uid'] = str(uuid.uuid4())
    async with _reader(world, loop_env) as reader:
        got = await reader.root_of(
            ['a-car-0000', 'b-truck-0000', 'k00', 'a-review', 'b-truck-0002', 'missing']
        )

    assert got == {'a-car-0000': 'r10001', 'b-truck-0000': 'r10002', 'k00': 'r4'}, (
        'a review member, an item in a cluster the partition lacks and a missing one are in no root'
    )


async def test_items_of_another_partition_are_not_in_this_ones_roots(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    async with _reader(world, loop_env) as reader:
        for crop_id in ('a-car-0000', 'a-car-0001'):
            world.doc(crop_id)['cluster_partition_id'] = str(uuid.uuid4())
        survey = await reader.survey(
            PartitionScope(cluster_uids=[str(world.uids[A])], min_root_size=1)
        )
        got = await reader.root_of(['a-car-0000', 'a-car-0002'])

    assert [s.members for s in survey] == [7]
    assert got == {'a-car-0002': 'r10001'}


async def test_a_reader_bound_to_another_partition_finds_none_of_the_items(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    world.serve(loop_env)
    async with _sessions(loop_env) as sessions:
        stranger = PartitionReader.for_run(
            world.os,  # type: ignore[arg-type]
            sessions,
            project=loop_env.project,
            pinned={'partition': {'partition_id': str(uuid.uuid4())}},
        )

        assert await stranger.survey(ALL) == []
        assert await stranger.root_of(['a-car-0000']) == {}


async def test_a_resumed_run_binds_to_the_partition_its_first_slice_pinned(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _world(monkeypatch)
    partition = world.serve(loop_env)
    async with _sessions(loop_env) as sessions:
        resumed = PartitionReader.for_run(
            world.os,  # type: ignore[arg-type]
            sessions,
            project=loop_env.project,
            pinned={'partition': {'kind': 'serving_partition', 'partition_id': str(partition)}},
        )
        first = PartitionReader.for_run(
            world.os,  # type: ignore[arg-type]
            sessions,
            project=loop_env.project,
            pinned={},
        )

        assert resumed.partition_id == partition
        assert first.partition_id is None
        assert await resumed.root_of(['a-car-0000']) == {'a-car-0000': 'r10001'}
        with pytest.raises(RuntimeError, match='no pinned partition'):
            await first.root_of(['a-car-0000'])


async def test_a_project_serving_no_partition_cannot_pin(loop_env: LoopEnv) -> None:
    async with _sessions(loop_env) as sessions:
        reader = PartitionReader(
            None,  # type: ignore[arg-type]
            sessions,
            project=loop_env.project,
        )

        with pytest.raises(NoServingPartitionError):
            await reader.pin()


async def test_a_root_snapshot_round_trips_through_the_blob_store(
    loop_env: LoopEnv, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = _world(monkeypatch)
    async with _reader(world, loop_env) as reader:
        (root,) = await reader.snapshots(
            PartitionScope(cluster_uids=[str(world.uids[A])], min_root_size=1)
        )
    store = SnapshotStore(SLUG, uuid.uuid4(), store=RootedStore({'projects': tmp_path}))

    key = await store.write_root(root)

    assert await store.read_root(key) == root
    with pytest.raises(ValueError, match='missing or unreadable'):
        await store.read_root(key.replace('r10001', 'r1'))


async def test_a_malformed_root_blob_is_an_error_never_an_empty_root(tmp_path: Path) -> None:
    store = SnapshotStore(SLUG, uuid.uuid4(), store=RootedStore({'projects': tmp_path}))
    key = await store.write_ids('nodes', 'r1', ['a'])

    with pytest.raises(ValueError, match='malformed'):
        await store.read_root(key)

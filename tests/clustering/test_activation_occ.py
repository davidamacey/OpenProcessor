"""Activation is optimistic: a stale expected partition or revision is 409, never a silent win."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from _config_fixtures import make_project
from sqlalchemy import text

from src.db.session import async_session_factory, project_tx
from src.services.clustering.activation import (
    PartitionConflictError,
    activate,
    complete_projection,
    get_activation,
)
from src.services.clustering.partitions import (
    PartitionMember,
    PartitionSpace,
    mint_uids,
    record_partition,
)


pytestmark = [pytest.mark.db, pytest.mark.asyncio]
SPACE = PartitionSpace('pe_embedding', 'pe_core@v1', 'pe_core_v1')


@pytest.fixture
def project(pg) -> uuid.UUID:
    return make_project(pg, f'ac-{uuid.uuid4().hex[:10]}')


async def _partition(project) -> uuid.UUID:
    async with async_session_factory()() as session, project_tx(session, project):
        (uid,) = await mint_uids(session, 1)
        return await record_partition(
            session,
            project=project,
            target='item',
            level='class',
            space=SPACE,
            method='ivf',
            params={},
            members=[PartitionMember('a', uid), PartitionMember('b', uid)],
        )


async def _activate(project, partition, serving, revision, by='test'):
    async with async_session_factory()() as session, project_tx(session, project):
        return await activate(
            session,
            project=project,
            partition_id=partition,
            expected_serving=serving,
            expected_revision=revision,
            by=by,
        )


async def _state(project):
    async with async_session_factory()() as session, project_tx(session, project):
        return await get_activation(
            session, project=project, target='item', level='class', embedding_field='pe_embedding'
        )


async def test_a_stale_expected_partition_is_409(pg_env, project) -> None:
    first, second, third = [await _partition(project) for _ in range(3)]
    await _activate(project, first, None, None)
    async with async_session_factory()() as session, project_tx(session, project):
        assert await complete_projection(session, project=project, partition_id=first)
    state = await _state(project)
    assert state is not None
    assert state.serving_partition_id == first
    await _activate(project, second, first, state.revision)
    with pytest.raises(PartitionConflictError):  # the caller still believes `first` is serving
        await _activate(project, third, first, state.revision)
    with pytest.raises(PartitionConflictError):  # right partition, stale revision
        await _activate(project, third, first, state.revision - 1)


async def test_two_racing_activations_have_one_winner(pg_env, project) -> None:
    first, a, b = [await _partition(project) for _ in range(3)]
    await _activate(project, first, None, None)
    state = await _state(project)
    assert state is not None
    results = await asyncio.gather(
        _activate(project, a, first, state.revision),
        _activate(project, b, first, state.revision),
        return_exceptions=True,
    )
    assert sorted(type(r).__name__ for r in results) == ['Activation', 'PartitionConflictError']


async def test_one_activation_writes_one_projection_task_and_one_event(pg_env, project) -> None:
    partition = await _partition(project)
    await _activate(project, partition, None, None)
    async with async_session_factory()() as session, project_tx(session, project):
        tasks = (
            await session.execute(
                text(
                    "SELECT count(*) FROM search_projection_tasks WHERE kind = 'cluster_partition'"
                )
            )
        ).scalar_one()
        events = (
            await session.execute(text('SELECT count(*) FROM cluster_activation_events'))
        ).scalar_one()
    assert (tasks, events) == (1, 1)


async def test_a_not_ready_or_foreign_partition_is_refused(pg_env, pg, project) -> None:
    other = make_project(pg, f'ac2-{uuid.uuid4().hex[:10]}')
    foreign = await _partition(other)
    from src.services.clustering.activation import PartitionNotReadyError

    with pytest.raises(PartitionNotReadyError):
        await _activate(project, foreign, None, None)

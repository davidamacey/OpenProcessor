"""An older projection cannot overwrite a newer partition's fields; a single activation projects once."""

from __future__ import annotations

import uuid

import pytest
from _config_fixtures import make_project
from sqlalchemy import text

from src.db.session import async_session_factory, project_tx
from src.services.clustering.activation import activate, get_activation
from src.services.clustering.partitions import (
    PartitionMember,
    PartitionSpace,
    mint_uids,
    record_partition,
)
from src.services.clustering.projection import project_partition, update_action
from tests.clustering._fake_opensearch import FakeOpenSearch


pytestmark = [pytest.mark.db, pytest.mark.asyncio]
SPACE = PartitionSpace('pe_embedding', 'pe_core@v1', 'pe_core_v1')


@pytest.fixture
def project(pg) -> uuid.UUID:
    return make_project(pg, f'pj-{uuid.uuid4().hex[:10]}')


async def _partition(project, uid_for_b=None) -> tuple[uuid.UUID, list[uuid.UUID]]:
    async with async_session_factory()() as session, project_tx(session, project):
        uids = await mint_uids(session, 2)
        partition = await record_partition(
            session,
            project=project,
            target='item',
            level='class',
            space=SPACE,
            method='ivf',
            params={},
            members=[
                PartitionMember('a', uids[0], 'assigned', 0.9, 0.5),
                PartitionMember('b', uids[1], 'assigned', 0.8, 0.4),
                PartitionMember('c', None, 'unassigned'),
            ],
        )
        return partition, uids


async def _activate(project, partition, serving=None, revision=None):
    async with async_session_factory()() as session, project_tx(session, project):
        await activate(
            session,
            project=project,
            partition_id=partition,
            expected_serving=serving,
            expected_revision=revision,
            by='test',
        )


async def _project(project, partition, fake, old=None):
    async with async_session_factory()() as session, project_tx(session, project):
        return await project_partition(
            session,
            fake,
            project=project,
            partition_id=partition,
            index='items',
            target='item',
            old_partition_id=old,
        )


async def test_a_single_activation_projects_once_and_flips_serving(pg_env, project) -> None:
    partition, uids = await _partition(project)
    await _activate(project, partition)
    fake = FakeOpenSearch()
    result = await _project(project, partition, fake)
    assert result.status == 'flipped'
    assert fake.calls == 1
    assert fake.docs['a']['cluster_uid'] == str(uids[0])
    assert fake.docs['c']['cluster_state'] == 'unassigned'
    async with async_session_factory()() as session, project_tx(session, project):
        state = await get_activation(
            session, project=project, target='item', level='class', embedding_field='pe_embedding'
        )
        done = (
            await session.execute(
                text('SELECT count(*) FROM search_projection_tasks WHERE done_at IS NOT NULL')
            )
        ).scalar_one()
    assert state is not None
    assert (state.serving_partition_id, state.pending_partition_id) == (partition, None)
    assert done == 1


async def test_an_older_projection_writes_nothing_after_a_newer_activation(pg_env, project) -> None:
    first, _ = await _partition(project)
    second, _ = await _partition(project)
    await _activate(project, first)
    fake = FakeOpenSearch()
    assert (await _project(project, first, fake)).status == 'flipped'
    await _activate(project, second, first, 1)
    # a slow projection of `first` now finds that `second` is pending: it must not write
    stale = FakeOpenSearch()
    result = await _project(project, first, stale)
    assert result.status == 'superseded'
    assert stale.calls == 0
    assert (await _project(project, second, fake, old=first)).status == 'flipped'
    assert fake.docs['a']['cluster_partition_id'] == str(second)


async def test_a_document_already_holding_a_newer_partition_is_left_alone(pg_env, project) -> None:
    first, _ = await _partition(project)
    second, _ = await _partition(project)
    assert str(second) > str(first)  # UUIDv7: text order is creation order
    await _activate(project, first)
    fake = FakeOpenSearch()
    fake.docs['a'] = {'cluster_partition_id': str(second), 'cluster_uid': 'newer'}
    await _project(project, first, fake)
    assert fake.docs['a']['cluster_uid'] == 'newer'
    assert fake.docs['b']['cluster_partition_id'] == str(first)


async def test_failed_items_keep_the_old_partition_serving(pg_env, project) -> None:
    partition, _ = await _partition(project)
    await _activate(project, partition)
    result = await _project(project, partition, FakeOpenSearch(fail_ids={'b'}))
    assert (result.status, result.failed) == ('failed', 1)
    async with async_session_factory()() as session, project_tx(session, project):
        state = await get_activation(
            session, project=project, target='item', level='class', embedding_field='pe_embedding'
        )
    assert state is not None
    assert state.pending_partition_id == partition


def test_a_box_key_updates_inside_the_item() -> None:
    action, body = update_action(
        'items',
        'region_box',
        'crop1:boxA',
        uuid.uuid4(),
        {'uid': None, 'state': 'unassigned', 'score': None, 'margin': None},
    )
    assert action['update']['_id'] == 'crop1'
    assert body['script']['params']['box_id'] == 'boxA'

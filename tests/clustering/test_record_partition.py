"""``record_partition`` writes a whole partition in one transaction, memberships by COPY."""

from __future__ import annotations

import uuid

import psycopg
import pytest
from _config_fixtures import make_project
from sqlalchemy import text

from src.db.session import async_session_factory, core_tx, project_tx
from src.services.clustering.partitions import (
    PartitionError,
    PartitionMember,
    PartitionSpace,
    mint_uids,
    record_partition,
)


pytestmark = [pytest.mark.db, pytest.mark.asyncio]

SPACE = PartitionSpace('pe_embedding', 'pe_core@v1', 'pe_core_v1')


@pytest.fixture
def project(pg) -> uuid.UUID:
    return make_project(pg, f'rp-{uuid.uuid4().hex[:10]}')


def _members(uids: list[uuid.UUID], n: int) -> list[PartitionMember]:
    out = [
        PartitionMember(f'crop{i:06d}', uids[i % len(uids)], 'assigned', 0.9, 0.1) for i in range(n)
    ]
    out.append(PartitionMember('lonely', None, 'unassigned'))
    return out


async def test_ten_thousand_members_write_in_one_copy_and_are_ready(pg_env, project) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        uids = await mint_uids(session, 20)
        partition = await record_partition(
            session,
            project=project,
            target='item',
            level='class',
            space=SPACE,
            method='ivf',
            params={'n_clusters': 20},
            members=_members(uids, 10_000),
        )
    async with async_session_factory()() as session, project_tx(session, project):
        status, n_items, n_clusters, n_un = (
            await session.execute(
                text(
                    'SELECT status, n_items, n_clusters, n_unassigned FROM cluster_partitions '
                    'WHERE id = :id'
                ),
                {'id': partition},
            )
        ).one()
        memberships = (
            await session.execute(
                text(
                    'SELECT count(*), count(project_id) FROM cluster_memberships WHERE partition_id = :id'
                ),
                {'id': partition},
            )
        ).one()
        sizes = (
            await session.execute(
                text(
                    'SELECT sum(size), min(display_no), max(display_no) FROM partition_clusters WHERE partition_id = :id'
                ),
                {'id': partition},
            )
        ).one()
        identities = (
            await session.execute(
                text('SELECT count(*) FROM cluster_identities WHERE first_partition_id = :id'),
                {'id': partition},
            )
        ).scalar_one()
    assert (status, n_items, n_clusters, n_un) == ('ready', 10_001, 20, 1)
    assert memberships == (10_001, 10_001)
    assert sizes == (10_000, 1, 20)
    assert identities == 20


async def test_a_ready_partition_is_immutable_afterwards(pg_env, pg, project) -> None:
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
            members=_members(uids, 6),
        )
    with (
        psycopg.connect(pg.conninfo('super'), autocommit=True) as conn,
        pytest.raises(psycopg.errors.IntegrityConstraintViolation),
    ):
        conn.execute(
            'UPDATE cluster_memberships SET score = 0.1 WHERE partition_id = %s', (partition,)
        )


async def test_a_bad_member_writes_nothing(pg_env, pg, project) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        uids = await mint_uids(session, 1)
        bad = [PartitionMember('a', uids[0], 'unassigned')]
        with pytest.raises(PartitionError, match='unassigned iff'):
            await record_partition(
                session,
                project=project,
                target='item',
                level='class',
                space=SPACE,
                method='ivf',
                params={},
                members=bad,
            )
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        assert conn.execute(
            'SELECT count(*) FROM cluster_partitions WHERE project_id = %s', (project,)
        ).fetchone() == (0,)


async def test_a_face_partition_is_core_scoped(pg_env, project) -> None:
    async with async_session_factory()() as session, core_tx(session, 'test'):
        uids = await mint_uids(session, 2)
        partition = await record_partition(
            session,
            project=None,
            target='face',
            level='identity',
            space=PartitionSpace('face_embedding', 'arcface@v1', 'arcface'),
            method='union_find',
            params={'threshold': 0.6},
            members=_members(uids, 8),
        )
    async with async_session_factory()() as session, project_tx(session, project):
        seen = (
            await session.execute(
                text('SELECT count(*) FROM cluster_partitions WHERE id = :id'), {'id': partition}
            )
        ).scalar_one()
    assert seen == 0  # a project scope never sees a core partition

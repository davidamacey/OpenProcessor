"""The loop's roots carry (partition_id, cluster_uid, display_no) and never an integer cluster id."""

from __future__ import annotations

import ast
import inspect
from typing import TYPE_CHECKING

import pytest
from fixtures.job_rows import make_project

from src.db.session import async_session_factory, project_tx
from src.services.clustering.activation import activate
from src.services.clustering.partitions import (
    PartitionMember,
    PartitionSpace,
    mint_uids,
    record_partition,
)
from src.services.curation.assist_loop import partition as partition_module
from src.services.curation.assist_loop.partition import ClusterRoot, PartitionReader


if TYPE_CHECKING:
    import uuid

pytestmark = [pytest.mark.db, pytest.mark.asyncio]
SPACE = PartitionSpace('pe_embedding', 'pe_core@v1', 'pe_core_v1')


@pytest.fixture
def project(pg) -> uuid.UUID:
    return make_project(pg)


async def _serve(project) -> tuple[uuid.UUID, list[uuid.UUID]]:
    async with async_session_factory()() as session, project_tx(session, project):
        uids = await mint_uids(session, 2)
        members = [PartitionMember(f'c{i}', uids[0], 'assigned') for i in range(5)]
        members += [PartitionMember(f'r{i}', uids[0], 'review') for i in range(2)]
        members += [PartitionMember(f'd{i}', uids[1], 'assigned') for i in range(3)]
        partition = await record_partition(
            session,
            project=project,
            target='item',
            level='class',
            space=SPACE,
            method='ivf',
            params={},
            members=members,
        )
        await activate(
            session,
            project=project,
            partition_id=partition,
            expected_serving=None,
            expected_revision=None,
            by='test',
        )
    return partition, uids


async def test_the_roots_are_the_serving_partitions_clusters(pg_env, project) -> None:
    partition, uids = await _serve(project)
    async with async_session_factory()() as session, project_tx(session, project):
        roots = await PartitionReader.roots(session, project=project, target='item')
    assert roots == [
        ClusterRoot(partition, uids[0], 7, display_no=1),
        ClusterRoot(partition, uids[1], 3, display_no=2),
    ]


async def test_a_project_with_no_serving_partition_has_no_roots(pg_env, project) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        assert await PartitionReader.roots(session, project=project, target='item') == []


def test_the_reader_names_no_integer_cluster_id() -> None:
    source = inspect.getsource(PartitionReader) + inspect.getsource(ClusterRoot)
    names = {n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)}
    assert 'cluster_id' not in names
    assert 'cluster_id' not in source
    assert partition_module.ClusterRoot is ClusterRoot

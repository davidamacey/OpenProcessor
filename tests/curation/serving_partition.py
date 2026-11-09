"""Give the bound project a serving item partition with known clusters, for the move tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.project_context import current_project_id
from src.db.session import async_session_factory, project_tx
from src.services.clustering.activation import activate
from src.services.clustering.partitions import (
    PartitionMember,
    PartitionSpace,
    mint_uids,
    record_partition,
)


if TYPE_CHECKING:
    import uuid

SPACE = PartitionSpace('pe_embedding', 'pe_core@v1', 'pe_core_v1')


async def serve_clusters(n_clusters: int = 2) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """Record and activate an item partition with ``n_clusters`` clusters of two members each.

    Returns ``(partition_id, cluster_uids)`` in display order.
    """
    project = current_project_id()
    async with async_session_factory()() as session, project_tx(session, project):
        uids = await mint_uids(session, n_clusters)
        members = [
            PartitionMember(f'member-{i}-{j}', uid, 'assigned')
            for i, uid in enumerate(uids)
            for j in range(2)
        ]
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

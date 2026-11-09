"""Does a cluster exist in the partition the project is serving?"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from src.services.clustering.assign import serving_partition_id


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession


async def serving_cluster_partition(
    session: AsyncSession, *, project: uuid.UUID, target: str, cluster_uid: uuid.UUID
) -> uuid.UUID | None:
    """The serving partition of ``target`` when it holds ``cluster_uid``, else ``None`` (no
    serving partition, or the cluster belongs to another partition). Call inside ``project_tx``."""
    partition_id = await serving_partition_id(session, project=project, target=target)
    if partition_id is None:
        return None
    row = await session.execute(
        text('SELECT 1 FROM partition_clusters WHERE partition_id = :p AND cluster_uid = :u'),
        {'p': partition_id, 'u': cluster_uid},
    )
    return partition_id if row.first() is not None else None

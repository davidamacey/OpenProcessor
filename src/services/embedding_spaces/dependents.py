"""Queue the artefacts that read the primary space only, after a flip (embeddings plan 5.5).

Clustering and near-duplicate scores are computed from the primary space's vectors, so a flip
makes both stale for that target. The flip is already committed when these start, so a start that
is refused because the same job is already active (409 ``job_busy``) is not an error: the active
job is the one that will pick the new primary up on its next resolve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.exceptions import RequestRejectedError
from src.domain.embedding_spaces import Target
from src.services.clustering.job import start_run as start_cluster_run
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos import embedding_spaces as repo

_CLUSTER_TARGET = {Target.ITEM: 'item', Target.BOX: 'region_box'}
_NEAR_DUP_LEVEL = {Target.ITEM: 'item', Target.FRAME: 'image'}


def _busy(exc: RequestRejectedError) -> bool:
    return isinstance(exc.detail, dict) and exc.detail.get('error') == 'job_busy'


async def queue_dependents(
    session: AsyncSession, project_id: uuid.UUID, primary: repo.SpaceRow, *, requested_by: str
) -> list[str]:
    """Job kinds queued for the new primary of ``primary.target``; empty when it has no dependents."""
    queued: list[str] = []
    cluster_target = _CLUSTER_TARGET.get(primary.target)
    if cluster_target is not None:
        try:
            await start_cluster_run(
                session,
                project_id,
                target=cluster_target,
                level='class',
                embedding_field=primary.vector_field,
                requested_by=requested_by,
            )
            queued.append('cluster_run')
        except RequestRejectedError as exc:
            if not _busy(exc):
                raise
    level = _NEAR_DUP_LEVEL.get(primary.target)
    if level is not None:
        try:
            await start_job(
                session,
                'near_dup_score',
                project_id,
                params={'level': level},
                requested_by=requested_by,
            )
            queued.append('near_dup_score')
        except RequestRejectedError as exc:
            if not _busy(exc):
                raise
    return queued


__all__ = ['queue_dependents']

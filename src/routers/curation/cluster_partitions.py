"""Curation router sub-module: cluster runs, partitions and activation (plan 5.5, S-core).

``POST clusters/runs`` queues a ``cluster_run``; ``GET clusters/partitions`` lists what runs
produced; ``PUT clusters/activation`` makes one partition pending (409 ``partition_conflict`` when
the caller's view is stale) and queues the projection that serves it.
"""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves the path parameter type
from typing import Annotated

from fastapi import Query

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.core.exceptions import RequestRejectedError
from src.db.session import project_tx
from src.domain.embedding_spaces import Target
from src.routers.curation._common import router
from src.schemas.curation.cluster_partitions import (
    ClusterActivationAccepted,
    ClusterActivationRequest,
    ClusterPartitionList,
    ClusterPartitionView,
    ClusterRunAccepted,
    ClusterRunRequest,
)
from src.services.clustering import activation as activation_service
from src.services.clustering.job import start_projection, start_run
from src.services.clustering.partitions import PartitionRow, get_partition, list_partitions
from src.services.embedding_spaces.resolve import embedding_space


_ACTOR = 'api'
_TARGETS = {'item': Target.ITEM, 'region_box': Target.BOX}


def _view(row: PartitionRow) -> ClusterPartitionView:
    return ClusterPartitionView(
        **{
            **row.__dict__,
            'metrics': dict(row.metrics),
            'diff': None if row.diff is None else dict(row.diff),
        }
    )


@router.post('/clusters/runs', response_model=ClusterRunAccepted, status_code=202)
async def start_cluster_run(session: DbSessionDep, body: ClusterRunRequest) -> ClusterRunAccepted:
    """Queue a clustering run for a target; 409 `job_busy` while one for that space is active."""
    space = await embedding_space(_TARGETS[body.target])
    started = await start_run(
        session,
        current_project_id(),
        target=body.target,
        level=body.level,
        embedding_field=space.vector_field,
        requested_by=_ACTOR,
    )
    return ClusterRunAccepted(job_id=str(started.job.id))


@router.get('/clusters/partitions', response_model=ClusterPartitionList)
async def list_cluster_partitions(
    session: DbSessionDep,
    target: Annotated[str | None, Query(pattern='^(item|region_box)$')] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ClusterPartitionList:
    async with project_tx(session, current_project_id()):
        rows = await list_partitions(session, target=target, limit=limit)
    return ClusterPartitionList(partitions=[_view(r) for r in rows])


@router.get('/clusters/partitions/{partition_id}', response_model=ClusterPartitionView)
async def get_cluster_partition(
    session: DbSessionDep, partition_id: uuid.UUID
) -> ClusterPartitionView:
    async with project_tx(session, current_project_id()):
        row = await get_partition(session, partition_id)
    if row is None:
        raise RequestRejectedError(404, {'error': 'partition_not_found'})
    return _view(row)


@router.put('/clusters/activation', response_model=ClusterActivationAccepted, status_code=202)
async def activate_cluster_partition(
    session: DbSessionDep, body: ClusterActivationRequest
) -> ClusterActivationAccepted:
    """Activate a ready partition; 409 `partition_conflict` if the serving partition or revision moved."""
    project = current_project_id()
    try:
        async with project_tx(session, project):
            activated = await activation_service.activate(
                session,
                project=project,
                partition_id=body.partition_id,
                expected_serving=body.expected_serving_partition_id,
                expected_revision=body.expected_revision,
                by=_ACTOR,
            )
    except activation_service.PartitionConflictError as exc:
        raise RequestRejectedError(
            409, {'error': 'partition_conflict', 'message': str(exc)}
        ) from exc
    except activation_service.PartitionNotReadyError as exc:
        raise RequestRejectedError(
            404, {'error': 'partition_not_ready', 'message': str(exc)}
        ) from exc
    started = await start_projection(
        session,
        project,
        partition_id=body.partition_id,
        target=activated.target,
        level=activated.level,
        embedding_field=activated.embedding_field,
        requested_by=_ACTOR,
    )
    return ClusterActivationAccepted(
        job_id=str(started.job.id),
        revision=activated.revision,
        pending_partition_id=body.partition_id,
    )

"""Curation router sub-module: embedding spaces.

``GET /curation/embedding/models`` is deployment-wide; the rest live under
``/curation/projects/{project}/embedding``. Register, build and retire answer 202 with the job that
works on the space; pause uses the generic job route; cancel (with `abandon`) is the run route. Every refusal is the registry's
stable ``{"error": code, "message": ...}`` body.
"""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves the path parameter type
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Header, Query

from src.config.curation_accessors import images_index, items_index
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.core.dependencies import get_curation_opensearch
from src.core.exceptions import RequestRejectedError
from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.routers.curation._common import router
from src.routers.curation._embedding_space_models import (
    EmbeddingBuildRequest,
    EmbeddingModelEntry,
    EmbeddingModelListResponse,
    EmbeddingPromoteRequest,
    EmbeddingPromoteResponse,
    EmbeddingReindexRunResponse,
    EmbeddingRunCancelResponse,
    EmbeddingSpaceListResponse,
    EmbeddingSpaceRegisterRequest,
    EmbeddingSpaceStartResponse,
    EmbeddingSpaceView,
)
from src.routers.curation.projects import global_router
from src.services.embedding_spaces import flip, spaces, start, views
from src.services.embedding_spaces.capacity import OpenSearchCapacity, SpaceCapacity
from src.services.embedding_spaces.dependents import queue_dependents
from src.services.embedding_spaces.reindex import ReindexIndex, ReindexMode
from src.services.embedding_spaces.reindex_opensearch import OpenSearchReindexIndex
from src.services.jobs.service import rejection


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

IdempotencyKey = Annotated[
    str | None,
    Header(
        alias='Idempotency-Key', description='8-200 characters; a repeat answers the first job.'
    ),
]
TargetQuery = Annotated[Target, Query(description='The target of the space (item, box, frame).')]
REQUESTER = 'api'


@asynccontextmanager
async def _refusals() -> AsyncIterator[None]:
    """A registry refusal is answered with its own status and stable code."""
    try:
        yield
    except EmbeddingSpaceError as exc:
        raise RequestRejectedError(exc.status, {'error': exc.code, 'message': exc.detail}) from exc


async def _capacity() -> SpaceCapacity:
    return OpenSearchCapacity(await get_curation_opensearch(), items_index(), images_index())


CapacityDep = Annotated[SpaceCapacity, Depends(_capacity)]


async def _index() -> ReindexIndex:
    return OpenSearchReindexIndex(await get_curation_opensearch())


IndexDep = Annotated[ReindexIndex, Depends(_index)]


def _view(found: views.SpaceWithModel, docs: int | None = None) -> EmbeddingSpaceView:
    row, model = found.space, found.model
    return EmbeddingSpaceView(
        space_key=row.space_key,
        target=row.target,
        model_key=model.model_key,
        model_version=model.version,
        dim=row.dim,
        role=row.role.value,
        status=row.status.value,
        ready_at=row.ready_at,
        docs_with_vector=docs,
    )


@global_router.get(
    '/embedding/models', response_model=EmbeddingModelListResponse, tags=['Embedding']
)
async def list_embedding_models(session: DbSessionDep) -> EmbeddingModelListResponse:
    """The deployment's embedding catalog with whether each version can back a space now."""
    return EmbeddingModelListResponse(
        models=[
            EmbeddingModelEntry(
                model_key=e.model.model_key,
                version=e.model.version,
                modalities=list(e.model.modalities),
                native_dim=e.model.native_dim,
                matryoshka_dims=list(e.model.matryoshka_dims),
                licence=e.model.licence,
                available=e.availability.available,
                unavailable_reason=e.availability.reason,
            )
            for e in await views.catalog_entries(session)
        ]
    )


@router.get('/embedding/spaces', response_model=EmbeddingSpaceListResponse, tags=['Embedding'])
async def list_embedding_spaces(
    session: DbSessionDep, index: IndexDep, target: Annotated[Target | None, Query()] = None
) -> EmbeddingSpaceListResponse:
    """The project's spaces (live and retired), by target."""
    project = current_project_id()
    rows = await spaces.list_project_spaces(session, project, target)
    found = await views.with_models(session, project, rows)
    return EmbeddingSpaceListResponse(
        spaces=[_view(f, await views.docs_with_vector(index, f.space)) for f in found]
    )


def _started(result: start.Started) -> EmbeddingSpaceStartResponse:
    return EmbeddingSpaceStartResponse(
        space_key=result.space.space_key,
        target=result.space.target,
        job_id=None if result.job_id is None else str(result.job_id),
        run_id=None if result.run_id is None else str(result.run_id),
        bytes_estimate=result.bytes_estimate,
    )


@router.post(
    '/embedding/spaces',
    response_model=EmbeddingSpaceStartResponse,
    status_code=202,
    tags=['Embedding'],
)
async def register_embedding_space(
    body: EmbeddingSpaceRegisterRequest,
    session: DbSessionDep,
    capacity: CapacityDep,
    idempotency_key: IdempotencyKey = None,
) -> EmbeddingSpaceStartResponse:
    """Register a candidate space and start its build. 409 `embedding_space_exists`, 422
    `embedding_model_unavailable`, 409 `embedding_insufficient_storage` when the data path is short."""
    async with _refusals():
        result = await start.register_and_build(
            session,
            current_project_id(),
            capacity,
            model_key=body.model_key,
            version=body.model_version,
            target=body.target,
            dim=body.dim,
            idempotency_key=idempotency_key,
            requested_by=REQUESTER,
        )
    return _started(result)


@router.post(
    '/embedding/spaces/{space_key}/build',
    response_model=EmbeddingSpaceStartResponse,
    status_code=202,
    tags=['Embedding'],
)
async def build_embedding_space(
    space_key: str,
    target: TargetQuery,
    body: EmbeddingBuildRequest,
    session: DbSessionDep,
    capacity: CapacityDep,
    idempotency_key: IdempotencyKey = None,
) -> EmbeddingSpaceStartResponse:
    """Re-start a failed or cancelled build, or backfill documents a ready space lacks. 409
    `job_busy` while another job works on the space."""
    async with _refusals():
        result = await start.build_again(
            session,
            current_project_id(),
            capacity,
            target=target,
            key=space_key,
            mode=ReindexMode(body.mode),
            batch_size=body.batch_size,
            idempotency_key=idempotency_key,
            requested_by=REQUESTER,
        )
    return _started(result)


@router.post(
    '/embedding/spaces/{space_key}/promote',
    response_model=EmbeddingPromoteResponse,
    tags=['Embedding'],
)
async def promote_embedding_space(
    space_key: str, target: TargetQuery, body: EmbeddingPromoteRequest, session: DbSessionDep
) -> EmbeddingPromoteResponse:
    """The fenced flip. 409 `embedding_space_not_ready` or `embedding_evaluation_missing`."""
    project = current_project_id()
    async with _refusals():
        result = await flip.promote(
            session, project, target=target, key=space_key, force=body.force
        )
    await queue_dependents(session, project, result.primary, requested_by=REQUESTER)
    rows = [result.primary] + ([result.previous] if result.previous else [])
    shown = [_view(f) for f in await views.with_models(session, project, rows)]
    return EmbeddingPromoteResponse(
        primary=shown[0],
        previous=shown[1] if result.previous else None,
        config_epoch=result.config_epoch,
    )


@router.post(
    '/embedding/spaces/{space_key}/retire',
    response_model=EmbeddingSpaceStartResponse,
    status_code=202,
    tags=['Embedding'],
)
async def retire_embedding_space(
    space_key: str, target: TargetQuery, session: DbSessionDep
) -> EmbeddingSpaceStartResponse:
    """Retire an optional space and queue its `clear` run. The primary cannot be retired."""
    async with _refusals():
        result = await start.retire_and_clear(
            session, current_project_id(), target=target, key=space_key, requested_by=REQUESTER
        )
    return _started(result)


@router.get(
    '/embedding/reindex-runs/{run_id}',
    response_model=EmbeddingReindexRunResponse,
    tags=['Embedding'],
)
async def get_embedding_reindex_run(
    run_id: uuid.UUID, session: DbSessionDep
) -> EmbeddingReindexRunResponse:
    """The run with its job's status and progress. 404 for another project's run."""
    project = current_project_id()
    found = await views.run_view(session, project, run_id)
    if found is None:
        raise rejection(404, 'job_not_found', f'no re-index run {run_id}')
    return EmbeddingReindexRunResponse(**found)


@router.post(
    '/embedding/reindex-runs/{run_id}/cancel',
    response_model=EmbeddingRunCancelResponse,
    tags=['Embedding'],
)
async def cancel_embedding_reindex_run(
    run_id: uuid.UUID,
    session: DbSessionDep,
    abandon: Annotated[
        bool, Query(description='Mark the building space failed and clear its partial vectors.')
    ] = False,
) -> EmbeddingRunCancelResponse:
    """Cancel a run. Without `abandon` the space stays `building` (re-startable as a backfill)."""
    async with _refusals():
        done = await start.cancel_run(
            session, current_project_id(), run_id, abandon=abandon, requested_by=REQUESTER
        )
    return EmbeddingRunCancelResponse(
        run_id=str(done.run_id),
        job_status=done.job_status,
        abandoned=done.abandoned,
        space_status=done.space_status,
        clear_job_id=None if done.clear_job_id is None else str(done.clear_job_id),
    )

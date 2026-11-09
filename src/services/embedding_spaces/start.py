"""Start the jobs of a space: register + build, build (or backfill) again, retire + clear.

The job service is the only way a ``jobs`` row is born; this module adds the space rule in front
of it and the run row beside it (``start_job(with_rows=...)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.db.repos import embedding_spaces as repo
from src.db.session import project_tx
from src.domain.embedding_spaces import EmbeddingSpaceError, Status, Target, space_key
from src.domain.job_states import TERMINAL
from src.services.embedding_spaces import reindex_runs, spaces
from src.services.embedding_spaces.capacity import SpaceCapacity, require_room
from src.services.embedding_spaces.reindex import EmbeddingReindexParams, ReindexMode
from src.services.jobs.service import cancel_job, start_job


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

KIND = 'embedding_reindex'


@dataclass(frozen=True)
class Started:
    space: repo.SpaceRow
    job_id: uuid.UUID | None
    run_id: uuid.UUID | None
    bytes_estimate: int | None
    created: bool = True


async def _start(
    session: AsyncSession,
    project_id: uuid.UUID,
    space: repo.SpaceRow,
    mode: ReindexMode,
    *,
    batch_size: int,
    estimate: int | None,
    idempotency_key: str | None,
    requested_by: str,
) -> Started:
    params = EmbeddingReindexParams(
        space_key=space.space_key, target=space.target, mode=mode, batch_size=batch_size
    ).model_dump(mode='json')
    run_ids: list[uuid.UUID] = []

    async def rows(inner: AsyncSession, job: object) -> None:
        run_ids.append(
            await reindex_runs.create_run(
                inner,
                project_id=project_id,
                space_id=space.id,
                job_id=job.id,  # type: ignore[attr-defined]
                mode=mode.value,
                bytes_estimate=estimate,
            )
        )

    result = await start_job(
        session,
        KIND,
        project_id,
        params=params,
        idempotency_key=idempotency_key,
        requested_by=requested_by,
        with_rows=rows,
    )
    return Started(
        space, result.job.id, run_ids[0] if run_ids else None, estimate, created=result.created
    )


async def register_and_build(
    session: AsyncSession,
    project_id: uuid.UUID,
    capacity: SpaceCapacity,
    *,
    model_key: str,
    version: int,
    target: Target,
    dim: int | None,
    idempotency_key: str | None,
    requested_by: str,
    batch_size: int = 64,
) -> Started:
    """Register a ``pending`` candidate space and queue its ``build``.

    A repeat with the same ``Idempotency-Key`` answers the first job (``created`` false) instead
    of ``embedding_space_exists``; a different key or none is that 409.
    """
    try:
        row = await spaces.register_space(
            session,
            project_id,
            model_key=model_key,
            version=version,
            target=target,
            dim=dim,
            created_by=requested_by,
        )
    except EmbeddingSpaceError as refusal:
        if refusal.code != 'embedding_space_exists' or idempotency_key is None:
            raise
        return await _repeat(
            session,
            project_id,
            model_key=model_key,
            version=version,
            target=target,
            dim=dim,
            idempotency_key=idempotency_key,
            batch_size=batch_size,
            requested_by=requested_by,
        )
    try:
        estimate = await require_room(capacity, target, row.dim)
    except EmbeddingSpaceError:
        await spaces.retire_space(session, project_id, target=target, key=row.space_key)
        raise
    return await _start(
        session,
        project_id,
        row,
        ReindexMode.BUILD,
        batch_size=batch_size,
        estimate=estimate,
        idempotency_key=idempotency_key,
        requested_by=requested_by,
    )


async def _repeat(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    model_key: str,
    version: int,
    target: Target,
    dim: int | None,
    idempotency_key: str,
    batch_size: int,
    requested_by: str,
) -> Started:
    """The space exists: it is a repeat only if the job service knows this key for these params."""
    async with project_tx(session, project_id):
        model = await session.run_sync(lambda s: repo.get_model_version(s, model_key, version))
        assert model is not None  # register_space found it
        row = await repo.get_space(
            session,
            project_id,
            target,
            space_key(model_key, version, dim or model.native_dim, model.native_dim),
        )
    assert row is not None
    return await _start(
        session,
        project_id,
        row,
        ReindexMode.BUILD,
        batch_size=batch_size,
        estimate=None,
        idempotency_key=idempotency_key,
        requested_by=requested_by,
    )


async def build_again(
    session: AsyncSession,
    project_id: uuid.UUID,
    capacity: SpaceCapacity,
    *,
    target: Target,
    key: str,
    mode: ReindexMode,
    batch_size: int,
    idempotency_key: str | None,
    requested_by: str,
) -> Started:
    """Re-start a ``failed`` or ``building`` space (``build``) or fill a ``ready`` one (``backfill``).

    A second job for the same space is 409 ``job_busy`` from the job's exclusive scope.
    """
    async with project_tx(session, project_id):
        row = await repo.get_space(session, project_id, target, key)
    if row is None:
        raise EmbeddingSpaceError(
            'embedding_space_not_found', f'no {target.value} space {key}', status=404
        )
    if row.status not in (Status.PENDING, Status.FAILED, Status.BUILDING, Status.READY):
        raise EmbeddingSpaceError(
            'embedding_space_state', f'{key} is {row.status.value}; it cannot be built'
        )
    estimate = await require_room(capacity, target, row.dim)
    return await _start(
        session,
        project_id,
        row,
        mode,
        batch_size=batch_size,
        estimate=estimate,
        idempotency_key=idempotency_key,
        requested_by=requested_by,
    )


async def retire_and_clear(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    target: Target,
    key: str,
    requested_by: str,
) -> Started:
    """``-> retiring`` and queue the ``clear`` run; a space that never held vectors retires at once."""
    row = await spaces.retire_space(session, project_id, target=target, key=key)
    if row.status is Status.RETIRED:
        return Started(row, None, None, None)
    return await _start(
        session,
        project_id,
        row,
        ReindexMode.CLEAR,
        batch_size=1,
        estimate=None,
        idempotency_key=None,
        requested_by=requested_by,
    )


@dataclass(frozen=True)
class Cancelled:
    run_id: uuid.UUID
    job_status: str
    abandoned: bool
    space_status: str
    clear_job_id: uuid.UUID | None


async def cancel_run(
    session: AsyncSession,
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    *,
    abandon: bool,
    requested_by: str,
) -> Cancelled:
    """Cancel a run's job; with ``abandon`` the building space is marked ``failed`` and cleared.

    A queued job is cancelled at once, so its clear is queued here. A running job stops at its
    next batch boundary and clears the field itself (it still holds the space's exclusive scope).
    """
    async with project_tx(session, project_id):
        ref = await reindex_runs.get_run_ref(session, run_id)
        if ref is None:
            raise EmbeddingSpaceError(
                'embedding_run_not_found', f'no re-index run {run_id}', status=404
            )
        space = await repo.get_space(session, project_id, Target(ref.target), ref.space_key)
    assert space is not None
    if abandon and (ref.mode != ReindexMode.BUILD.value or space.status is not Status.BUILDING):
        raise EmbeddingSpaceError(
            'embedding_space_state',
            f'{ref.space_key} is {space.status.value}; only a building space can be abandoned',
        )
    moved = await cancel_job(session, ref.job_id, project_id, requested_by=requested_by)
    if not abandon:
        return Cancelled(run_id, moved.job.status, False, space.status.value, None)
    failed = await spaces.mark_failed(session, project_id, target=space.target, key=ref.space_key)
    clear_job = None
    if moved.job.status in TERMINAL:
        clear_job = (
            await _start(
                session,
                project_id,
                failed,
                ReindexMode.CLEAR,
                batch_size=1,
                estimate=None,
                idempotency_key=None,
                requested_by=requested_by,
            )
        ).job_id
    return Cancelled(run_id, moved.job.status, True, failed.status.value, clear_job)

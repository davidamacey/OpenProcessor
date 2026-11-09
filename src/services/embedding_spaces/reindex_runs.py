"""The ``embedding_reindex_runs`` row of one re-index job: created with the job, counted by it."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, update

from src.db.models.embeddings import EmbeddingReindexRun, EmbeddingSpace
from src.db.models.jobs import Job


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession


async def create_run(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    space_id: uuid.UUID,
    job_id: uuid.UUID,
    mode: str,
    scope: Mapping[str, Any] | None = None,
    bytes_estimate: int | None = None,
) -> uuid.UUID:
    """Insert the run inside the job's start transaction (``start_job(with_rows=...)``)."""
    run = EmbeddingReindexRun(
        project_id=project_id,
        space_id=space_id,
        job_id=job_id,
        mode=mode,
        scope=dict(scope or {}),
        bytes_estimate=bytes_estimate,
    )
    session.add(run)
    await session.flush()
    return run.id


async def record_counts(
    session: AsyncSession,
    job_id: uuid.UUID,
    *,
    total: int | None,
    done: int,
    failed: int,
    skipped: int,
) -> None:
    """Idempotent: the counts are absolute, so a redelivered batch writes the same numbers."""
    started = dt.datetime.now(dt.UTC)
    await session.execute(
        update(EmbeddingReindexRun)
        .where(EmbeddingReindexRun.job_id == job_id)
        .values(
            docs_total=total,
            docs_done=done,
            docs_failed=failed,
            docs_skipped=skipped,
            started_at=func.coalesce(EmbeddingReindexRun.started_at, started),
        )
    )


async def finish_run(session: AsyncSession, job_id: uuid.UUID) -> None:
    await session.execute(
        update(EmbeddingReindexRun)
        .where(EmbeddingReindexRun.job_id == job_id)
        .values(finished_at=dt.datetime.now(dt.UTC))
    )


@dataclass(frozen=True)
class RunRef:
    job_id: uuid.UUID
    space_id: uuid.UUID
    space_key: str
    target: str
    mode: str


async def get_run_ref(session: AsyncSession, run_id: uuid.UUID) -> RunRef | None:
    found = (
        await session.execute(
            select(
                EmbeddingReindexRun.job_id,
                EmbeddingSpace.id,
                EmbeddingSpace.space_key,
                EmbeddingSpace.target,
                EmbeddingReindexRun.mode,
            )
            .join(EmbeddingSpace, EmbeddingSpace.id == EmbeddingReindexRun.space_id)
            .where(EmbeddingReindexRun.id == run_id)
        )
    ).one_or_none()
    return None if found is None else RunRef(*found)


async def get_run_view(session: AsyncSession, run_id: uuid.UUID) -> dict[str, Any] | None:
    """The run with its space key and its job's status and progress (row security scopes it)."""
    found = (
        await session.execute(
            select(EmbeddingReindexRun, EmbeddingSpace.space_key, Job.status, Job.progress)
            .join(EmbeddingSpace, EmbeddingSpace.id == EmbeddingReindexRun.space_id)
            .join(Job, Job.id == EmbeddingReindexRun.job_id)
            .where(EmbeddingReindexRun.id == run_id)
        )
    ).one_or_none()
    if found is None:
        return None
    run, key, status, progress = found
    return {
        'id': str(run.id),
        'space_key': key,
        'mode': run.mode,
        'scope': run.scope,
        'docs_total': run.docs_total,
        'docs_done': run.docs_done,
        'docs_failed': run.docs_failed,
        'docs_skipped': run.docs_skipped,
        'bytes_estimate': run.bytes_estimate,
        'started_at': run.started_at,
        'finished_at': run.finished_at,
        'job_id': str(run.job_id),
        'job_status': status,
        'job_progress': progress or {},
    }

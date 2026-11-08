"""The ``embedding_reindex_runs`` row of one re-index job: created with the job, counted by it."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, update

from src.db.models.embeddings import EmbeddingReindexRun


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

"""The combine run's rows: ``combine_runs`` and ``combine_sources`` beside its ``jobs`` row.

The job parameters hold the request; ``combine_runs.class_mapping`` holds the previewed plan
(target classes, per-source mapping, duplicates, originals). Combine routes are global, so the
lookups below run in deployment scope and then address the job in its target's scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.db.models.datasets import CombineRun, CombineSource
from src.db.session import deployment_tx


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.config.projects import ProjectRecord
    from src.db.repos.jobs import JobRow
    from src.services.projects.combine.models import HoldoutMode

_HOLDOUT_COLUMN: dict[str, str] = {
    'preserve_union': 'union',
    'recompute': 'recompute',
    'none': 'drop',
}


@dataclass(frozen=True)
class CombineRunRow:
    job_id: uuid.UUID
    target_project_id: uuid.UUID
    document: dict[str, Any]
    sources: list[str]


def holdout_column(mode: HoldoutMode) -> str:
    return _HOLDOUT_COLUMN[mode]


async def insert_rows(
    session: AsyncSession,
    job: JobRow,
    *,
    target: ProjectRecord,
    sources: list[ProjectRecord],
    holdout: HoldoutMode,
    document: dict[str, Any],
) -> None:
    """The run and its sources, in the transaction that inserts the job (``start_job`` ``with_rows``)."""
    assert target.id is not None
    session.add(
        CombineRun(
            job_id=job.id,
            target_project_id=target.id,
            holdout_mode=holdout_column(holdout),
            class_mapping=document,
        )
    )
    await session.flush()
    session.add_all(
        CombineSource(job_id=job.id, position=position, source_project_id=r.id, source_slug=r.slug)
        for position, r in enumerate(sources)
    )
    await session.flush()


async def find(session: AsyncSession, wanted: uuid.UUID) -> CombineRunRow | None:
    """The run of job ``wanted``; ``None`` when there is no such run."""
    async with deployment_tx(session, 'a combine route finds the target project of a job'):
        run = (
            await session.execute(select(CombineRun).where(CombineRun.job_id == wanted))
        ).scalar_one_or_none()
        if run is None:
            return None
        slugs = (
            await session.execute(
                select(CombineSource.source_slug)
                .where(CombineSource.job_id == wanted)
                .order_by(CombineSource.position)
            )
        ).scalars()
        return CombineRunRow(
            run.job_id, run.target_project_id, dict(run.class_mapping), list(slugs)
        )

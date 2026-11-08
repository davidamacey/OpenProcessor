"""``bakeoff_runs``: the row that belongs to a ``bakeoff`` job.

Functions take the caller's sync session inside its transaction and never commit. The start
service inserts the row in the job's own transaction (``start_job(with_rows=...)``); the handler
records where the results are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import update

from src.db.models.training import BakeoffRun


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


def insert_run(
    session: Session,
    *,
    job_id: uuid.UUID,
    project_id: uuid.UUID,
    eval_dataset_keys: Sequence[str],
    models: Sequence[Any],
) -> None:
    """One row per job; ``eval_dataset_key`` joins the compared datasets' ids with commas."""
    session.add(
        BakeoffRun(
            job_id=job_id,
            project_id=project_id,
            eval_dataset_key=','.join(eval_dataset_keys),
            models=list(models),
        )
    )
    session.flush()


def set_results_key(session: Session, job_id: uuid.UUID, key: str) -> bool:
    """Record where the results are; false when the job has no row."""
    result = session.execute(
        update(BakeoffRun).where(BakeoffRun.job_id == job_id).values(results_storage_key=key)
    )
    return bool(result.rowcount)  # type: ignore[attr-defined]

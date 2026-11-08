"""``bakeoff_runs``: the row that belongs to a ``bakeoff`` job.

Functions take the caller's sync session inside its transaction and never commit. The start
service inserts the row in the job's own transaction (``start_job(with_rows=...)``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

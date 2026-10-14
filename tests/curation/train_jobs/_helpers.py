"""Helpers shared by the training-job service tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.services.training.job_models import TrainJobStatus


if TYPE_CHECKING:
    from pathlib import Path


def _write_status(
    jobs_dir: Path,
    job_id: str,
    *,
    state: str = 'running',
    heartbeat_at: str | None = None,
) -> None:
    """Helper: write a status.json for a job_id."""
    status = TrainJobStatus(
        job_id=job_id,
        state=state,  # type: ignore[arg-type]
        heartbeat_at=heartbeat_at,
    )
    (jobs_dir / f'{job_id}.status.json').write_text(
        json.dumps(status.model_dump(mode='json')),
    )

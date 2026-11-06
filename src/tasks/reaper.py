"""``op.periodic.reaper``: stuck-job recovery and retention, every 30 s from beat (02b section 4)."""

from __future__ import annotations

from src.services.jobs.reaper import reap
from src.tasks.app import app


@app.task(
    name='op.periodic.reaper', shared=False, soft_time_limit=120, time_limit=150, ignore_result=True
)
def reaper() -> None:
    """One tick; idempotent, safe to overlap (advisory lock and ``SKIP LOCKED``)."""
    reap()

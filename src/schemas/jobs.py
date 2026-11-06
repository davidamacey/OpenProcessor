"""Wire models of the job runtime."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.domain.job_perf import JobPerfSummary


JobErrorCode = Literal[
    'job_busy',
    'job_not_found',
    'job_finished',
    'job_not_pausable',
    'job_cancelling',
    'resume_blocked',
    'idempotency_key_reused',
    'idempotency_key_invalid',
    'queue_unavailable',
    'unknown_job_kind',
    'invalid_cursor',
]


class JobView(BaseModel):
    """One job as the job routes show it: the row, without the lease internals."""

    id: str
    kind: str
    project: str | None = Field(description='Project slug; null for a deployment-wide job.')
    status: Literal[
        'queued',
        'running',
        'paused',
        'cancelling',
        'completed',
        'completed_with_errors',
        'failed',
        'cancelled',
        'interrupted',
    ]
    scope: str
    queue: str
    attempt: int
    retry_count: int
    max_attempts: int
    pause_reason: str | None
    pause_requested: bool
    cancel_requested: bool
    progress: dict[str, Any]
    progress_updated_at: dt.datetime | None
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    params: dict[str, Any] | None = Field(
        default=None, description='Only on the single-job route; the list leaves it out.'
    )
    requested_by: str
    created_at: dt.datetime
    queued_at: dt.datetime | None
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    updated_at: dt.datetime


class JobPage(BaseModel):
    """A page of jobs, newest first. Send ``next_cursor`` back as ``cursor`` for the next page."""

    items: list[JobView]
    next_cursor: str | None = None


__all__ = ['JobErrorCode', 'JobPage', 'JobPerfSummary', 'JobView']

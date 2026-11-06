"""Wire models of the job runtime."""

from __future__ import annotations

from typing import Literal

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
]

__all__ = ['JobErrorCode', 'JobPerfSummary']

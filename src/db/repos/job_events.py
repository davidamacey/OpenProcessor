"""The ``events`` rows a job's life produces: ``job.state`` and ``job.progress`` snapshots.

Written in the caller's transaction next to the ``jobs`` update that caused them, so a rolled-back
change emits nothing; the caller sends the Valkey wake-up after its commit
(``src.tasks.runtime.event_wakeup.wake``). Payloads are full snapshots under the table's
7,000-byte cap, never deltas (02b section 6.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.db.repos.events import EventPayloadTooLargeError, insert_event


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.orm import Session

    from src.db.repos.jobs import JobRow

STATE = 'job.state'
PROGRESS = 'job.progress'
_MESSAGE_CHARS = 500


def state_payload(job: JobRow) -> dict[str, Any]:
    return {
        'job_id': str(job.id),
        'kind': job.kind,
        'status': job.status,
        'queue': job.queue,
        'attempt': job.attempt,
        'retry_count': job.retry_count,
        'pause_reason': job.pause_reason,
        'pause_requested': job.pause_requested_at is not None,
        'cancel_requested': job.cancel_requested_at is not None,
        'error_code': job.error_code,
        'error_message': (job.error_message or '')[:_MESSAGE_CHARS] or None,
    }


def emit_state(session: Session, job: JobRow) -> None:
    insert_event(session, job.project_id, STATE, state_payload(job), job_id=job.id)


def emit_progress(session: Session, job: JobRow, snapshot: Mapping[str, Any]) -> None:
    """A ``job.progress`` snapshot; counters that would break the size cap are dropped first."""
    payload = {'job_id': str(job.id), 'kind': job.kind, 'progress': dict(snapshot)}
    try:
        insert_event(session, job.project_id, PROGRESS, payload, job_id=job.id)
    except EventPayloadTooLargeError:
        trimmed = {k: v for k, v in snapshot.items() if k in ('done', 'total', 'stage')}
        payload['progress'] = trimmed
        insert_event(session, job.project_id, PROGRESS, payload, job_id=job.id)

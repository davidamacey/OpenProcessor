"""The statements of the reaper (02b section 4): what is stuck, and the one move that fixes it.

Every select takes ``FOR UPDATE SKIP LOCKED`` ("multiple consumers accessing a queue-like table",
https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE), so two reapers never
act on one row, and every move names the status it expects, so a row that changed in between is
left alone. All comparisons use the database clock. Functions take the caller's sync session
inside its deployment transaction and never commit.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import column, delete, exists, func, select, table, text
from sqlalchemy.exc import IntegrityError

from src.db.models.jobs import Job
from src.db.repos.jobs import UQ_ACTIVE, JobRow, _to_row
from src.domain import job_states as js


if TYPE_CHECKING:
    from datetime import timedelta

    from sqlalchemy.orm import Session

BATCH = 200
INTERRUPTED_SETTLE_S = 30
REDISPATCH_AFTER_S = 120


def _count(result: Any) -> int:
    return int(result.rowcount)


def try_tick_lock(session: Session, lock_class: int) -> bool:
    """``pg_try_advisory_xact_lock(class, 0)``: false when another tick holds it."""
    return bool(
        session.execute(
            text('SELECT pg_try_advisory_xact_lock(:class, 0)'), {'class': lock_class}
        ).scalar_one()
    )


# --- rule a: a lease that expired ---------------------------------------------------------------

_LOSE = text(
    'WITH expired AS (SELECT id FROM jobs '
    "WHERE status IN ('running', 'cancelling') AND lease_expires_at < now() "
    'ORDER BY lease_expires_at LIMIT :batch FOR UPDATE SKIP LOCKED) '
    "UPDATE jobs j SET status = CASE WHEN j.status = 'cancelling' THEN 'cancelled' "
    "ELSE 'interrupted' END, "
    "finished_at = CASE WHEN j.status = 'cancelling' THEN now() END, "
    'owner = NULL, lease_expires_at = NULL, updated_at = now(), '
    "retry_count = j.retry_count + CASE WHEN j.status = 'running' THEN 1 ELSE 0 END "
    'FROM expired WHERE j.id = expired.id RETURNING j.*'
)


def lose_expired(session: Session, batch: int = BATCH) -> list[JobRow]:
    """``cancelling -> cancelled``, ``running -> interrupted``; the attempt ends ``lost``."""
    rows = [_to_row(r) for r in session.execute(_LOSE, {'batch': batch}).mappings()]
    for row in rows:
        session.execute(
            text(
                "UPDATE job_attempts SET ended_at = now(), outcome = 'lost' "
                'WHERE job_id = :id AND attempt = :attempt AND ended_at IS NULL'
            ),
            {'id': row.id, 'attempt': row.attempt},
        )
    return rows


# --- rule b: an interrupted job that resumes by itself ---------------------------------------

_SELECT_RESUMABLE = text(
    'SELECT j.id, j.retry_count >= j.max_attempts AS exhausted FROM jobs j '
    "WHERE j.status = 'interrupted' AND j.auto_resume "
    'AND j.updated_at < now() - make_interval(secs => :settle) '
    'AND (j.retry_count >= j.max_attempts OR NOT EXISTS (SELECT 1 FROM jobs a '
    'WHERE a.project_id IS NOT DISTINCT FROM j.project_id AND a.exclusive_group = j.exclusive_group '
    'AND a.scope = j.scope AND a.status = ANY(:active))) '
    'ORDER BY j.updated_at LIMIT :batch FOR UPDATE OF j SKIP LOCKED'
)


@dataclass(frozen=True)
class Resumable:
    id: uuid.UUID
    exhausted: bool


def select_resumable(session: Session, batch: int = BATCH) -> list[Resumable]:
    found = session.execute(
        _SELECT_RESUMABLE,
        {'settle': INTERRUPTED_SETTLE_S, 'active': sorted(js.ACTIVE), 'batch': batch},
    )
    return [Resumable(uuid.UUID(str(r.id)), r.exhausted) for r in found]


def fail_exhausted(session: Session, job_id: uuid.UUID) -> JobRow | None:
    """``interrupted -> failed`` (``attempts_exhausted``): the transition table allows it."""
    row = (
        session.execute(
            text(
                "UPDATE jobs SET status = 'failed', error_code = 'attempts_exhausted', "
                "error_message = 'every attempt ended in a failure', finished_at = now(), "
                "updated_at = now() WHERE id = :id AND status = 'interrupted' RETURNING *"
            ),
            {'id': job_id},
        )
        .mappings()
        .first()
    )
    return None if row is None else _to_row(row)


def requeue(session: Session, job_id: uuid.UUID, from_status: str) -> JobRow | None:
    """``interrupted`` or ``paused`` back to ``queued``, in a savepoint.

    A ``unique_violation`` from a racing start of the same group skips this row (it stays where it
    is, logged once by the caller) instead of rolling back the batch (review R-5); returns ``None``
    then, and also when the row moved on.
    """
    try:
        with session.begin_nested():
            row = (
                session.execute(
                    text(
                        "UPDATE jobs SET status = 'queued', pause_requested_at = NULL, "
                        'pause_reason = NULL, queued_at = now(), last_dispatched_at = NULL, '
                        'updated_at = now() WHERE id = :id AND status = :from_status RETURNING *'
                    ),
                    {'id': job_id, 'from_status': from_status},
                )
                .mappings()
                .first()
            )
    except IntegrityError as exc:
        if getattr(getattr(exc.orig, 'diag', None), 'constraint_name', None) != UQ_ACTIVE:
            raise
        return None
    return None if row is None else _to_row(row)


# --- rule c: queued and not dispatched, rule d: queued too long -------------------------------

_SELECT_UNDISPATCHED = text(
    "SELECT * FROM jobs WHERE status = 'queued' AND (last_dispatched_at IS NULL "
    'OR last_dispatched_at < now() - make_interval(secs => :after)) '
    'ORDER BY queued_at LIMIT :batch FOR UPDATE SKIP LOCKED'
)


def select_undispatched(session: Session, batch: int = BATCH) -> list[JobRow]:
    found = session.execute(
        _SELECT_UNDISPATCHED, {'after': REDISPATCH_AFTER_S, 'batch': batch}
    ).mappings()
    return [_to_row(r) for r in found]


_FAIL_QUEUE_WAIT = text(
    "WITH late AS (SELECT id FROM jobs WHERE status = 'queued' "
    'AND queued_at + max_queue_wait < now() ORDER BY queued_at LIMIT :batch '
    'FOR UPDATE SKIP LOCKED) '
    "UPDATE jobs j SET status = 'failed', error_code = 'queue_wait_exceeded', "
    "error_message = 'the job waited in the queue longer than its limit', "
    'finished_at = now(), updated_at = now() FROM late WHERE j.id = late.id RETURNING j.*'
)


def fail_queue_wait_exceeded(session: Session, batch: int = BATCH) -> list[JobRow]:
    return [_to_row(r) for r in session.execute(_FAIL_QUEUE_WAIT, {'batch': batch}).mappings()]


# --- rules e and i: projection tasks with no job ------------------------------------------------


@dataclass(frozen=True)
class ProjectionTask:
    id: uuid.UUID
    project_id: uuid.UUID
    kind: str
    payload: Any


_SELECT_PROJECTION_TASKS = text(
    'SELECT id, project_id, kind, payload FROM search_projection_tasks '
    'WHERE done_at IS NULL AND job_id IS NULL AND kind = ANY(:kinds) '
    'ORDER BY created_at LIMIT :batch FOR UPDATE SKIP LOCKED'
)


def select_projection_tasks(
    session: Session, kinds: list[str], batch: int = BATCH
) -> list[ProjectionTask]:
    found = session.execute(_SELECT_PROJECTION_TASKS, {'kinds': kinds, 'batch': batch})
    return [ProjectionTask(r.id, r.project_id, r.kind, r.payload) for r in found]


def link_projection_task(session: Session, task_id: uuid.UUID, job_id: uuid.UUID) -> None:
    session.execute(
        text('UPDATE search_projection_tasks SET job_id = :job WHERE id = :id AND job_id IS NULL'),
        {'job': job_id, 'id': task_id},
    )


# --- rule h: a system pause whose cause is gone ------------------------------------------------

_SELECT_SYSTEM_PAUSED = text(
    "SELECT j.* FROM jobs j WHERE j.status = 'paused' "
    "AND j.pause_reason IN ('project_paused', 'vlm_unavailable') "
    'ORDER BY j.updated_at LIMIT :batch FOR UPDATE OF j SKIP LOCKED'
)


def select_system_paused(session: Session, batch: int = BATCH) -> list[JobRow]:
    return [_to_row(r) for r in session.execute(_SELECT_SYSTEM_PAUSED, {'batch': batch}).mappings()]


# --- rule f: retention ----------------------------------------------------------------------


def _unreferenced_jobs_conditions(session: Session) -> list[Any]:
    """``NOT EXISTS`` over every foreign key into ``jobs`` that would refuse the delete.

    The tables that keep their job (``training_runs``, ``dataset_imports``, ``combine_runs``...)
    reference it without ``ON DELETE SET NULL``/``CASCADE``; they are read from the catalog so a
    table a later lane adds is covered without editing this rule.
    """
    refs = session.execute(
        text(
            'SELECT c.conrelid::regclass::text AS tbl, a.attname AS col FROM pg_constraint c '
            'JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey) '
            "WHERE c.contype = 'f' AND c.confrelid = 'jobs'::regclass "
            "AND c.confdeltype IN ('a', 'r') AND c.conrelid <> 'jobs'::regclass"
        )
    ).all()
    conditions = []
    for ref in refs:
        referencing = table(ref.tbl.rsplit('.', 1)[-1].strip('"'), column(ref.col))
        conditions.append(~exists().where(referencing.c[ref.col] == Job.id))
    return conditions


def delete_old_terminal_jobs(session: Session, max_age: timedelta, batch: int = BATCH) -> int:
    """Delete terminal rows older than ``max_age`` that no kept row still points at."""
    conditions = _unreferenced_jobs_conditions(session)
    ids = (
        select(Job.id)
        .where(
            Job.status.in_(sorted(js.TERMINAL)),
            Job.finished_at < func.now() - max_age,
            *conditions,
        )
        .order_by(Job.finished_at)
        .limit(batch)
        .with_for_update(skip_locked=True)
    )
    return _count(session.execute(delete(Job).where(Job.id.in_(ids))))


def delete_stale_workers(session: Session, max_age: timedelta) -> int:
    """``worker_capabilities`` not seen for ``max_age``."""
    return _count(
        session.execute(
            text('DELETE FROM worker_capabilities WHERE seen_at < now() - :age'), {'age': max_age}
        )
    )


def delete_stale_gpu_devices(session: Session, max_age: timedelta) -> int:
    """``gpu_devices`` not seen for ``max_age`` and held by no reservation."""
    return _count(
        session.execute(
            text(
                'DELETE FROM gpu_devices WHERE seen_at < now() - :age AND NOT EXISTS '
                '(SELECT 1 FROM gpu_reservations r WHERE r.gpu_uuid = gpu_devices.gpu_uuid)'
            ),
            {'age': max_age},
        )
    )


# --- rule g: stream worker claims ------------------------------------------------------------


def delete_expired_claims(session: Session) -> int:
    return _count(
        session.execute(text('DELETE FROM stream_worker_claims WHERE lease_expires_at < now()'))
    )

"""The ``jobs`` and ``job_attempts`` tables: start, claim, fence, finish, cancel, pause, resume.

Every function takes the caller's sync ``Session`` inside the caller's scoped transaction and
never commits; an async caller wraps one in ``AsyncSession.run_sync``. This is the only module
that writes ``jobs`` (``tests/test_jobs_single_entry.py``). The statements follow
``docs/design/v070_rebuild/02b_jobs_locking_events.md`` sections 1 to 4; the state machine itself
is ``src/domain/job_states.py`` and the database trigger built from it.

Fencing: a running job's ``fencing_token`` moves up by one at every claim, and every write the
holder makes (heartbeat, progress, checkpoint, the terminal write) names that token and the
live statuses in its ``WHERE``. A holder whose lease was taken over matches no row and gets
``None`` back (``Lost`` to the caller). ``retry_count`` grows only when an attempt ends
``interrupted``, ``lost`` or ``time_limit``; a pause, resume or yield never spends an attempt.

PostgreSQL references: the locking clause (https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE)
and READ COMMITTED (https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED):
an ``UPDATE`` that finds its row changed by a concurrent commit re-evaluates its ``WHERE`` on the
new version, which is what makes a single-statement claim a safe compare-and-set.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from sqlalchemy import insert, select, text
from sqlalchemy.exc import IntegrityError

from src.db.models.jobs import Job
from src.domain import job_states as js


if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Mapping

    from sqlalchemy.engine import RowMapping
    from sqlalchemy.orm import Session
    from sqlalchemy.sql.elements import TextClause

UQ_ACTIVE = 'uq_jobs_one_active'
UQ_IDEMPOTENCY = 'uq_jobs_idempotency'
# Outcomes that spend an attempt of the retry budget (02b section 1).
RETRY_OUTCOMES = frozenset({'interrupted', 'lost', 'time_limit'})
FINISH_STATUSES = frozenset({js.COMPLETED, js.COMPLETED_WITH_ERRORS, js.FAILED, js.CANCELLED})


class JobRepoError(Exception):
    """Base of the refusals this module raises; the service maps them to wire codes."""


class JobBusyError(JobRepoError):
    """Another job of the same exclusive group and scope is active (``uq_jobs_one_active``)."""

    def __init__(self, active_job_id: uuid.UUID | None) -> None:
        super().__init__(f'another job of this group is active: {active_job_id}')
        self.active_job_id = active_job_id


@dataclass(frozen=True)
class JobRow:
    """One ``jobs`` row as stored."""

    id: uuid.UUID
    project_id: uuid.UUID | None
    kind: str
    scope: str
    exclusive_group: str | None
    queue: str
    status: str
    params: Any
    checkpoint: Any | None
    progress: Any
    progress_updated_at: dt.datetime | None
    result: Any | None
    error_code: str | None
    error_message: str | None
    attempt: int
    retry_count: int
    max_attempts: int
    auto_resume: bool
    fencing_token: int
    owner: str | None
    heartbeat_at: dt.datetime | None
    lease_expires_at: dt.datetime | None
    max_queue_wait: dt.timedelta
    dispatch_count: int
    last_dispatched_at: dt.datetime | None
    cancel_requested_at: dt.datetime | None
    cancel_requested_by: str | None
    pause_requested_at: dt.datetime | None
    pause_reason: str | None
    idempotency_key: str | None
    parent_job_id: uuid.UUID | None
    requested_by: str
    requested_by_user_id: uuid.UUID | None
    created_at: dt.datetime
    queued_at: dt.datetime | None
    started_at: dt.datetime | None
    finished_at: dt.datetime | None
    updated_at: dt.datetime


_ROW_COLUMNS = [Job.__table__.c[f.name] for f in fields(JobRow)]


def _to_row(mapping: RowMapping) -> JobRow:
    return JobRow(**{f.name: mapping[f.name] for f in fields(JobRow)})


@dataclass(frozen=True)
class NewJob:
    """What the start path decides; the database assigns the id and the clocks."""

    kind: str
    project_id: uuid.UUID | None
    queue: str
    max_queue_wait: dt.timedelta
    requested_by: str
    scope: str = ''
    exclusive_group: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    auto_resume: bool = False
    max_attempts: int = 3
    idempotency_key: str | None = None
    parent_job_id: uuid.UUID | None = None
    requested_by_user_id: uuid.UUID | None = None


@dataclass(frozen=True)
class Inserted:
    """``created`` is false when an earlier row with the same idempotency key answered."""

    job: JobRow
    created: bool


def get(session: Session, job_id: uuid.UUID) -> JobRow | None:
    row = session.execute(select(*_ROW_COLUMNS).where(Job.id == job_id)).mappings().first()
    return None if row is None else _to_row(row)


def active_job_id(
    session: Session, project_id: uuid.UUID | None, group: str, scope: str
) -> uuid.UUID | None:
    """The id of the active job holding an exclusive group and scope, as the index sees it."""
    statement = text(
        'SELECT id FROM jobs WHERE project_id IS NOT DISTINCT FROM :project AND '
        'exclusive_group = :grp AND scope = :scope AND status = ANY(:active) LIMIT 1'
    )
    found = session.execute(
        statement,
        {'project': project_id, 'grp': group, 'scope': scope, 'active': sorted(js.ACTIVE)},
    ).scalar_one_or_none()
    return None if found is None else uuid.UUID(str(found))


def _by_idempotency(session: Session, new: NewJob) -> JobRow | None:
    row = (
        session.execute(
            select(*_ROW_COLUMNS).where(
                Job.project_id.is_not_distinct_from(new.project_id),
                Job.kind == new.kind,
                Job.idempotency_key == new.idempotency_key,
            )
        )
        .mappings()
        .first()
    )
    return None if row is None else _to_row(row)


def insert_queued(session: Session, new: NewJob) -> Inserted:
    """Insert a ``queued`` row inside a savepoint.

    A ``unique_violation`` aborts only the savepoint (PostgreSQL refuses every statement of an
    aborted transaction, https://www.postgresql.org/docs/current/sql-savepoint.html); after the
    rollback to it the same transaction reads the active job (``JobBusyError``) or, for a repeated
    idempotency key, the first job (``created=False``). A race with a committed delete of the
    holder is still reported busy with ``active_job_id=None``.
    """
    values = {
        'project_id': new.project_id,
        'kind': new.kind,
        'scope': new.scope,
        'exclusive_group': new.exclusive_group,
        'queue': new.queue,
        'status': js.QUEUED,
        'params': dict(new.params),
        'auto_resume': new.auto_resume,
        'max_attempts': new.max_attempts,
        'max_queue_wait': new.max_queue_wait,
        'idempotency_key': new.idempotency_key,
        'parent_job_id': new.parent_job_id,
        'requested_by': new.requested_by,
        'requested_by_user_id': new.requested_by_user_id,
        'queued_at': text('now()'),
    }
    try:
        with session.begin_nested():
            created = (
                session.execute(insert(Job).values(**values).returning(*_ROW_COLUMNS))
                .mappings()
                .one()
            )
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, 'diag', None), 'constraint_name', None)
        if constraint == UQ_IDEMPOTENCY:
            existing = _by_idempotency(session, new)
            if existing is not None:
                return Inserted(existing, created=False)
        if constraint == UQ_ACTIVE and new.exclusive_group is not None:
            raise JobBusyError(
                active_job_id(session, new.project_id, new.exclusive_group, new.scope)
            ) from exc
        raise
    return Inserted(_to_row(created), created=True)


def mark_dispatched(session: Session, job_id: uuid.UUID) -> None:
    session.execute(
        text(
            'UPDATE jobs SET dispatch_count = dispatch_count + 1, last_dispatched_at = now() '
            "WHERE id = :id AND status = 'queued'"
        ),
        {'id': job_id},
    )


def fail_queued(
    session: Session, job_id: uuid.UUID, error_code: str, error_message: str | None = None
) -> JobRow | None:
    """``queued -> failed`` (broker down at start, attempts exhausted, project unavailable...)."""
    row = (
        session.execute(
            text(
                "UPDATE jobs SET status = 'failed', error_code = :code, error_message = :msg, "
                'finished_at = now(), updated_at = now() '
                "WHERE id = :id AND status = 'queued' RETURNING *"
            ),
            {'id': job_id, 'code': error_code, 'msg': error_message},
        )
        .mappings()
        .first()
    )
    return None if row is None else _to_row(row)


# --- the wrapper's side -----------------------------------------------------------------------


@dataclass(frozen=True)
class JobHead:
    """What the wrapper reads under deployment scope before it knows the project."""

    id: uuid.UUID
    project_id: uuid.UUID | None
    kind: str
    project_status: str | None


def read_head(session: Session, job_id: uuid.UUID) -> JobHead | None:
    row = session.execute(
        text(
            'SELECT j.id, j.project_id, j.kind, p.status AS project_status FROM jobs j '
            'LEFT JOIN projects p ON p.id = j.project_id WHERE j.id = :id'
        ),
        {'id': job_id},
    ).first()
    return None if row is None else JobHead(row.id, row.project_id, row.kind, row.project_status)


@dataclass(frozen=True)
class Claimed:
    job: JobRow


@dataclass(frozen=True)
class ClaimRefused:
    """No row matched. ``reason`` is ``missing``, ``exhausted`` or ``not_queued`` (``status``)."""

    reason: str
    status: str | None = None


def claim(session: Session, job_id: uuid.UUID, owner: str, ttl_s: int) -> Claimed | ClaimRefused:
    """One statement admits one delivery: ``queued`` to ``running``, token and attempt up by one.

    Refuses a cancel-requested row and one whose retry budget is spent; a paused, cancelled,
    terminal, running or interrupted row is a stale or duplicate message and also matches
    nothing. The ``job_attempts`` row is inserted in the same transaction.
    """
    row = (
        session.execute(
            text(
                "UPDATE jobs SET status = 'running', attempt = attempt + 1, "
                'fencing_token = fencing_token + 1, owner = :owner, heartbeat_at = now(), '
                'lease_expires_at = now() + make_interval(secs => CAST(:ttl AS double precision)), '
                'started_at = coalesce(started_at, now()), updated_at = now() '
                "WHERE id = :id AND status = 'queued' AND cancel_requested_at IS NULL "
                'AND retry_count < max_attempts RETURNING *'
            ),
            {'id': job_id, 'owner': owner, 'ttl': ttl_s},
        )
        .mappings()
        .first()
    )
    if row is None:
        return _refusal(session, job_id)
    claimed = _to_row(row)
    session.execute(
        text(
            'INSERT INTO job_attempts (job_id, attempt, fencing_token, owner) '
            'VALUES (:id, :attempt, :token, :owner)'
        ),
        {
            'id': job_id,
            'attempt': claimed.attempt,
            'token': claimed.fencing_token,
            'owner': owner,
        },
    )
    return Claimed(claimed)


def _refusal(session: Session, job_id: uuid.UUID) -> ClaimRefused:
    row = session.execute(
        text('SELECT status, retry_count, max_attempts FROM jobs WHERE id = :id'), {'id': job_id}
    ).first()
    if row is None:
        return ClaimRefused('missing')
    if row.status == js.QUEUED and row.retry_count >= row.max_attempts:
        return ClaimRefused('exhausted', row.status)
    return ClaimRefused('not_queued', row.status)


@dataclass(frozen=True)
class Control:
    """What a fenced heartbeat, progress or checkpoint write learns about the row."""

    status: str
    cancel_requested: bool
    pause_requested: bool
    pause_reason: str | None


def _control(row: Any) -> Control | None:
    if row is None:
        return None
    return Control(row.status, row.cancel_requested, row.pause_requested, row.pause_reason)


def heartbeat(session: Session, job_id: uuid.UUID, token: int, ttl_s: int) -> Control | None:
    """Renew the lease; ``None`` means the token no longer holds the row (``Lost``).

    Touches only ``heartbeat_at`` and ``lease_expires_at``, which are in no index, so the update
    stays heap-only (``tests/db/test_jobs_hot_updates.py``).
    """
    row = session.execute(
        text(
            'UPDATE jobs SET heartbeat_at = now(), '
            'lease_expires_at = now() + make_interval(secs => CAST(:ttl AS double precision)) '
            "WHERE id = :id AND fencing_token = :token AND status IN ('running', 'cancelling') "
            'RETURNING status, cancel_requested_at IS NOT NULL AS cancel_requested, '
            'pause_requested_at IS NOT NULL AS pause_requested, pause_reason'
        ),
        {'id': job_id, 'token': token, 'ttl': ttl_s},
    ).first()
    return _control(row)


def progress(
    session: Session, job_id: uuid.UUID, token: int, snapshot: Mapping[str, Any]
) -> Control | None:
    """Fenced progress write (a full snapshot, not a delta). ``None``: ``Lost``."""
    row = session.execute(
        text(
            'UPDATE jobs SET progress = CAST(:progress AS jsonb), progress_updated_at = now() '
            "WHERE id = :id AND fencing_token = :token AND status IN ('running', 'cancelling') "
            'RETURNING status, cancel_requested_at IS NOT NULL AS cancel_requested, '
            'pause_requested_at IS NOT NULL AS pause_requested, pause_reason'
        ),
        {'id': job_id, 'token': token, 'progress': _json(snapshot)},
    ).first()
    return _control(row)


def checkpoint(
    session: Session, job_id: uuid.UUID, token: int, cursor: Mapping[str, Any] | None
) -> Control | None:
    """Fenced resume-cursor write. ``None``: ``Lost``."""
    row = session.execute(
        text(
            'UPDATE jobs SET checkpoint = CAST(:cursor AS jsonb) '
            "WHERE id = :id AND fencing_token = :token AND status IN ('running', 'cancelling') "
            'RETURNING status, cancel_requested_at IS NOT NULL AS cancel_requested, '
            'pause_requested_at IS NOT NULL AS pause_requested, pause_reason'
        ),
        {'id': job_id, 'token': token, 'cursor': None if cursor is None else _json(cursor)},
    ).first()
    return _control(row)


def request_pause(session: Session, job_id: uuid.UUID, token: int, reason: str) -> bool:
    """The handler asks for its own pause (``audit``, ``budget``...). ``False``: ``Lost``."""
    row = session.execute(
        text(
            'UPDATE jobs SET pause_requested_at = coalesce(pause_requested_at, now()), '
            'pause_reason = coalesce(pause_reason, :reason) '
            "WHERE id = :id AND fencing_token = :token AND status = 'running' RETURNING 1"
        ),
        {'id': job_id, 'token': token, 'reason': reason},
    ).first()
    return row is not None


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, default=str)


def _end_attempt(
    session: Session,
    row: JobRow,
    outcome: str,
    error_code: str | None,
    error_message: str | None,
) -> None:
    session.execute(
        text(
            'UPDATE job_attempts SET ended_at = now(), outcome = :outcome, '
            'error_code = :code, error_message = :msg WHERE job_id = :id AND attempt = :attempt'
        ),
        {
            'id': row.id,
            'attempt': row.attempt,
            'outcome': outcome,
            'code': error_code,
            'msg': error_message,
        },
    )


# The fenced moves: each names the token and the live statuses, releases the lease and adds
# :bump (1 for an outcome that spends a retry) to retry_count.
_FINISH = text(
    "UPDATE jobs SET status = CASE WHEN status = 'cancelling' THEN 'cancelled' ELSE :status END, "
    'result = CAST(:result AS jsonb), error_code = :code, error_message = :msg, '
    'finished_at = now(), owner = NULL, lease_expires_at = NULL, updated_at = now(), '
    'retry_count = retry_count + :bump '
    "WHERE id = :id AND fencing_token = :token AND status IN ('running', 'cancelling') "
    'RETURNING *'
)
_INTERRUPT = text(
    "UPDATE jobs SET status = CASE WHEN status = 'cancelling' THEN 'cancelled' "
    "ELSE 'interrupted' END, "
    "finished_at = CASE WHEN status = 'cancelling' THEN now() END, "
    'owner = NULL, lease_expires_at = NULL, updated_at = now(), retry_count = retry_count + :bump '
    "WHERE id = :id AND fencing_token = :token AND status IN ('running', 'cancelling') "
    'RETURNING *'
)
_PAUSE_RUNNING = text(
    "UPDATE jobs SET status = 'paused', pause_reason = coalesce(pause_reason, :reason, 'user'), "
    'owner = NULL, lease_expires_at = NULL, updated_at = now(), retry_count = retry_count + :bump '
    "WHERE id = :id AND fencing_token = :token AND status = 'running' RETURNING *"
)
_QUEUE_YIELD = text(
    "UPDATE jobs SET status = 'queued', queued_at = now(), last_dispatched_at = NULL, "
    'owner = NULL, lease_expires_at = NULL, updated_at = now(), retry_count = retry_count + :bump '
    "WHERE id = :id AND fencing_token = :token AND status = 'running' RETURNING *"
)


def _fenced_move(
    session: Session,
    statement: TextClause,
    job_id: uuid.UUID,
    token: int,
    *,
    outcome: str,
    params: Mapping[str, Any],
    error_code: str | None = None,
    error_message: str | None = None,
) -> JobRow | None:
    bump = 1 if outcome in RETRY_OUTCOMES else 0
    row = (
        session.execute(statement, {'id': job_id, 'token': token, 'bump': bump, **params})
        .mappings()
        .first()
    )
    if row is None:
        return None
    moved = _to_row(row)
    _end_attempt(session, moved, outcome, error_code, error_message)
    return moved


def finish(
    session: Session,
    job_id: uuid.UUID,
    token: int,
    *,
    status: str,
    outcome: str,
    result: Mapping[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> JobRow | None:
    """The fenced terminal write. ``None``: ``Lost``, nothing was written.

    A row that is ``cancelling`` can only end ``cancelled`` (the transition table has no other
    exit), so a handler that finishes or fails after a cancel request is recorded cancelled with
    its result or error kept.
    """
    if status not in FINISH_STATUSES:
        raise ValueError(f'not a terminal status: {status}')
    params = {
        'status': status,
        'result': None if result is None else _json(result),
        'code': error_code,
        'msg': error_message,
    }
    moved = _fenced_move(
        session,
        _FINISH,
        job_id,
        token,
        outcome=outcome,
        params=params,
        error_code=error_code,
        error_message=error_message,
    )
    if moved is not None and moved.status != status:
        _end_attempt(session, moved, 'cancelled', error_code, error_message)
    return moved


def interrupt(session: Session, job_id: uuid.UUID, token: int) -> JobRow | None:
    """Worker shutdown: ``running -> interrupted`` (``cancelling -> cancelled``); spends a retry."""
    return _fenced_move(session, _INTERRUPT, job_id, token, outcome='interrupted', params={})


def pause_running(
    session: Session, job_id: uuid.UUID, token: int, reason: str | None = None
) -> JobRow | None:
    """The handler stopped at a boundary after a pause request: ``running -> paused``."""
    return _fenced_move(
        session, _PAUSE_RUNNING, job_id, token, outcome='paused', params={'reason': reason}
    )


def queue_yield(session: Session, job_id: uuid.UUID, token: int) -> JobRow | None:
    """A sliced job gives way at a boundary: ``running -> queued``; no retry is spent."""
    return _fenced_move(session, _QUEUE_YIELD, job_id, token, outcome='yielded', params={})


# --- user controls ----------------------------------------------------------------------------


def cancel(session: Session, job_id: uuid.UUID, requested_by: str) -> JobRow | None:
    """Cancel in one statement (02b section 3.3); a database write only, never a Celery revoke.

    ``queued``, ``paused`` and ``interrupted`` end ``cancelled`` at once; ``running`` becomes
    ``cancelling`` and the wrapper writes ``cancelled``; ``cancelling`` stays as it is. ``None``
    when the row is missing or already terminal.
    """
    row = (
        session.execute(
            text(
                'UPDATE jobs SET cancel_requested_at = coalesce(cancel_requested_at, now()), '
                'cancel_requested_by = coalesce(cancel_requested_by, :by), '
                "status = CASE WHEN status IN ('queued', 'paused', 'interrupted') "
                "THEN 'cancelled' ELSE 'cancelling' END, "
                "finished_at = CASE WHEN status IN ('queued', 'paused', 'interrupted') "
                'THEN now() END, '
                "updated_at = CASE WHEN status IN ('queued', 'paused', 'interrupted') "
                'THEN now() ELSE updated_at END '
                "WHERE id = :id AND status IN ('queued', 'running', 'paused', 'interrupted', "
                "'cancelling') RETURNING *"
            ),
            {'id': job_id, 'by': requested_by},
        )
        .mappings()
        .first()
    )
    return None if row is None else _to_row(row)


def pause(session: Session, job_id: uuid.UUID, reason: str) -> JobRow | None:
    """A pause request (02b section 3.3): ``queued`` pauses at once, ``running`` is asked to.

    ``paused`` is returned unchanged. ``None`` when the row cannot be paused from where it is
    (missing, terminal, ``interrupted``, ``cancelling``); the caller reads it to tell which.
    """
    row = (
        session.execute(
            text(
                "UPDATE jobs SET status = CASE WHEN status = 'queued' THEN 'paused' ELSE status END, "
                'pause_requested_at = coalesce(pause_requested_at, now()), '
                'pause_reason = coalesce(pause_reason, :reason), '
                "updated_at = CASE WHEN status = 'queued' THEN now() ELSE updated_at END "
                "WHERE id = :id AND status IN ('queued', 'running', 'paused') "
                'RETURNING *'
            ),
            {'id': job_id, 'reason': reason},
        )
        .mappings()
        .first()
    )
    return None if row is None else _to_row(row)


def resume(session: Session, job_id: uuid.UUID) -> JobRow | None:
    """``paused`` or ``interrupted`` back to ``queued``; the caller dispatches after the commit.

    Runs in a savepoint: the singleton index may refuse the move when another job of the group
    became active meanwhile (``JobBusyError``). ``None``: the row is not resumable.
    """
    try:
        with session.begin_nested():
            row = (
                session.execute(
                    text(
                        "UPDATE jobs SET status = 'queued', pause_requested_at = NULL, "
                        'pause_reason = NULL, queued_at = now(), last_dispatched_at = NULL, '
                        'updated_at = now() '
                        "WHERE id = :id AND status IN ('paused', 'interrupted') "
                        'RETURNING *'
                    ),
                    {'id': job_id},
                )
                .mappings()
                .first()
            )
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, 'diag', None), 'constraint_name', None)
        if constraint != UQ_ACTIVE:
            raise
        held = get(session, job_id)
        busy = None
        if held is not None and held.exclusive_group is not None:
            busy = active_job_id(session, held.project_id, held.exclusive_group, held.scope)
        raise JobBusyError(busy) from exc
    return None if row is None else _to_row(row)

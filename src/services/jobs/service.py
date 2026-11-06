"""Start, cancel, pause and resume a job: the one write path the routes and starters call.

``start_job`` is the only way a ``jobs`` row is born (``tests/test_jobs_single_entry.py``). Its
order is fixed (02b section 2.2): insert the ``queued`` row in a savepoint inside the project's
transaction, commit, send the message, set the marker, record the dispatch. The message is sent
only after the commit, so a worker never receives an id it cannot read (trap T-9); a broker
failure marks the row ``failed`` and answers 503. A crash between the commit and the send leaves
a ``queued`` row with no dispatch, which reaper rule c sends.

Cancel, pause and resume are database writes. There is no Celery ``revoke`` anywhere (D-71): a
stale message finds the row ``cancelled``, ``paused`` or terminal at the claim and is
acknowledged.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.clients import task_queue
from src.config.job_kinds import KINDS
from src.core.exceptions import RequestRejectedError
from src.db.repos import job_events, jobs as repo
from src.db.session import deployment_tx, project_tx
from src.domain import job_states as js
from src.services.jobs import dispatch as dispatching
from src.services.jobs.blockers import resume_blocker
from src.tasks.runtime.event_wakeup import wake


if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Callable, Mapping
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session

    from src.config.job_kinds.kind import JobKind
    from src.schemas.jobs import JobErrorCode

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_MIN = 8
IDEMPOTENCY_KEY_MAX = 200
SYSTEM_REQUESTER = 'system'


def rejection(status: int, code: JobErrorCode, message: str, **fields: Any) -> RequestRejectedError:
    """The refusal the app answers as ``{"detail": {"error": code, "message": ...}}``."""
    return RequestRejectedError(status, {'error': code, 'message': message, **fields})


@dataclass(frozen=True)
class StartResult:
    """``created`` is false when the idempotency key matched an earlier start (answer 200)."""

    job: repo.JobRow
    created: bool


@dataclass(frozen=True)
class ControlResult:
    """The row after a cancel, pause or resume; ``changed`` is false for an idempotent repeat."""

    job: repo.JobRow
    changed: bool


def _kind(name: str) -> JobKind:
    try:
        return KINDS[name]
    except KeyError:
        raise rejection(500, 'unknown_job_kind', f'no job kind named {name!r}') from None


def _scope_value(kind: JobKind, params: Mapping[str, Any], scope: str | None) -> str:
    """The exclusive scope: the caller's explicit value, else the ``params`` key the kind names."""
    if scope is not None:
        return scope
    if not kind.exclusive_scope:
        return ''
    if kind.exclusive_scope not in params:
        raise ValueError(
            f'job kind {kind.name} needs params[{kind.exclusive_scope!r}] as its scope'
        )
    return str(params[kind.exclusive_scope])


def _check_project(kind: JobKind, project: uuid.UUID | None) -> None:
    if kind.project_scope in ('project', 'target') and project is None:
        raise ValueError(f'job kind {kind.name} needs a project')
    if kind.project_scope == 'deployment' and project is not None:
        raise ValueError(f'job kind {kind.name} is deployment-wide and takes no project')


def _check_key(key: str | None) -> None:
    if key is not None and not IDEMPOTENCY_KEY_MIN <= len(key) <= IDEMPOTENCY_KEY_MAX:
        raise rejection(
            422,
            'idempotency_key_invalid',
            f'Idempotency-Key must be {IDEMPOTENCY_KEY_MIN}-{IDEMPOTENCY_KEY_MAX} characters',
        )


def _same_params(stored: Any, given: Mapping[str, Any]) -> bool:
    return bool(stored == json.loads(json.dumps(dict(given), default=str)))


def insert_job(
    session: Session,
    kind: str,
    project: uuid.UUID | None,
    *,
    params: Mapping[str, Any] | None = None,
    scope: str | None = None,
    idempotency_key: str | None = None,
    requested_by: str = SYSTEM_REQUESTER,
    parent_job_id: uuid.UUID | None = None,
) -> StartResult:
    """The insert half of a start, inside the caller's transaction (no commit, no dispatch).

    Raises a 409 ``job_busy`` refusal naming the active job, or a 422 ``idempotency_key_reused``
    when the key was used before with other parameters. Reaper rules e and i call this inside
    their own deployment transaction; everything else calls :func:`start_job`.
    """
    spec = _kind(kind)
    given = dict(params or {})
    _check_project(spec, project)
    _check_key(idempotency_key)
    new = repo.NewJob(
        kind=spec.name,
        project_id=project,
        queue=spec.queue,
        max_queue_wait=dt.timedelta(seconds=spec.max_queue_wait_s),
        requested_by=requested_by,
        scope=_scope_value(spec, given, scope),
        exclusive_group=spec.exclusive_group,
        params=given,
        auto_resume=spec.auto_resume,
        max_attempts=spec.max_attempts,
        idempotency_key=idempotency_key,
        parent_job_id=parent_job_id,
    )
    try:
        inserted = repo.insert_queued(session, new)
    except repo.JobBusyError as busy:
        raise rejection(
            409,
            'job_busy',
            f'another {spec.exclusive_group} job is already active',
            active_job_id=None if busy.active_job_id is None else str(busy.active_job_id),
        ) from None
    if not inserted.created and not _same_params(inserted.job.params, given):
        raise rejection(
            422,
            'idempotency_key_reused',
            'this Idempotency-Key was used with different parameters',
        )
    if inserted.created:
        job_events.emit_state(session, inserted.job)
    return StartResult(inserted.job, inserted.created)


def _transaction(
    session: AsyncSession, project: uuid.UUID | None
) -> AbstractAsyncContextManager[AsyncSession]:
    if project is None:
        return deployment_tx(session, 'job control for a deployment-wide job')
    return project_tx(session, project)


async def start_job(
    session: AsyncSession,
    kind: str,
    project: uuid.UUID | None,
    *,
    params: Mapping[str, Any] | None = None,
    scope: str | None = None,
    idempotency_key: str | None = None,
    requested_by: str = SYSTEM_REQUESTER,
    parent_job_id: uuid.UUID | None = None,
    with_rows: Callable[[AsyncSession, repo.JobRow], Awaitable[None]] | None = None,
) -> StartResult:
    """Start a job and answer once it is queued (the route replies 202 with ``job.id``).

    ``with_rows`` inserts the domain rows that belong to the job (an import, a combine run) in
    the same transaction; it is not called for an idempotent repeat. The session must have no
    transaction in progress.
    """
    async with _transaction(session, project):
        result = await session.run_sync(
            lambda sync: insert_job(
                sync,
                kind,
                project,
                params=params,
                scope=scope,
                idempotency_key=idempotency_key,
                requested_by=requested_by,
                parent_job_id=parent_job_id,
            )
        )
        if result.created and with_rows is not None:
            await with_rows(session, result.job)
    wake(project)
    if not result.created:
        return result
    await _dispatch_after_commit(session, project, result.job)
    return result


async def _dispatch_after_commit(
    session: AsyncSession,
    project: uuid.UUID | None,
    job: repo.JobRow,
    *,
    fail_on_broker_error: bool = True,
) -> None:
    """Send the message and record the dispatch.

    A start whose broker refuses the message fails the row (the caller was told nothing yet); a
    resume leaves the row ``queued`` for reaper rule c, because the job existed before.
    """
    try:
        await asyncio.to_thread(dispatching.dispatch, job.kind, job.id)
    except task_queue.QueueUnavailable as exc:
        if not fail_on_broker_error:
            logger.warning('dispatch_deferred_to_reaper job_id=%s', job.id)
            return
        reason = str(exc)
        async with _transaction(session, project):
            failed = await session.run_sync(lambda sync: _fail_after_commit(sync, job.id, reason))
        wake(project)
        logger.error('queue_unavailable job_id=%s kind=%s', job.id, job.kind)
        raise rejection(
            503,
            'queue_unavailable',
            'the job queue is unavailable; retry shortly',
            job_id=str(job.id),
            job_status=None if failed is None else failed.status,
        ) from exc
    async with _transaction(session, project):
        await session.run_sync(lambda sync: repo.mark_dispatched(sync, job.id))


def _fail_after_commit(session: Session, job_id: uuid.UUID, message: str) -> repo.JobRow | None:
    failed = repo.fail_queued(session, job_id, 'queue_unavailable', message)
    if failed is not None:
        job_events.emit_state(session, failed)
    return failed


async def _locked(session: AsyncSession, job_id: uuid.UUID) -> repo.JobRow:
    row = await session.run_sync(lambda sync: repo.get(sync, job_id))
    if row is None:
        raise rejection(404, 'job_not_found', f'no job {job_id}')
    return row


async def cancel_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    project: uuid.UUID | None,
    *,
    requested_by: str,
) -> ControlResult:
    """Cancel (02b section 3.3): at once when nothing runs it, else ``cancelling``.

    409 ``job_finished`` for a terminal job; a repeat on ``cancelling`` changes nothing.
    """
    async with _transaction(session, project):
        before = await _locked(session, job_id)
        if before.status in js.TERMINAL:
            raise rejection(409, 'job_finished', f'job {job_id} already {before.status}')
        moved = await session.run_sync(lambda sync: repo.cancel(sync, job_id, requested_by))
        if moved is None:  # finished between the read and the write
            raise rejection(409, 'job_finished', f'job {job_id} already finished')
        changed = before.status != js.CANCELLING
        if changed:
            await session.run_sync(lambda sync: job_events.emit_state(sync, moved))
    wake(project)
    if before.status == js.QUEUED:
        await asyncio.to_thread(dispatching.forget_marker, job_id)
    return ControlResult(moved, changed)


async def pause_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    project: uuid.UUID | None,
    *,
    reason: str = 'user',
) -> ControlResult:
    """Pause a pausable kind: ``queued`` pauses at once, ``running`` is asked to (``pausing``)."""
    async with _transaction(session, project):
        before = await _locked(session, job_id)
        if not _kind(before.kind).pausable:
            raise rejection(409, 'job_not_pausable', f'{before.kind} jobs cannot be paused')
        if before.status == js.CANCELLING:
            raise rejection(409, 'job_cancelling', f'job {job_id} is being cancelled')
        if before.status in js.TERMINAL:
            raise rejection(409, 'job_finished', f'job {job_id} already {before.status}')
        moved = await session.run_sync(lambda sync: repo.pause(sync, job_id, reason))
        if moved is None:
            raise rejection(409, 'job_not_pausable', f'a {before.status} job cannot be paused')
        changed = moved.status != before.status or (
            before.pause_requested_at is None and moved.pause_requested_at is not None
        )
        if changed:
            await session.run_sync(lambda sync: job_events.emit_state(sync, moved))
    wake(project)
    if before.status == js.QUEUED and moved.status == js.PAUSED:
        await asyncio.to_thread(dispatching.forget_marker, job_id)
    return ControlResult(moved, changed)


async def resume_job(
    session: AsyncSession, job_id: uuid.UUID, project: uuid.UUID | None
) -> ControlResult:
    """``paused`` or ``interrupted`` back to ``queued`` under the same id, then dispatch.

    409 ``resume_blocked`` names the reason while the project is paused or the VLM is
    unavailable; 409 ``job_busy`` when another job of the group became active. A broker
    failure leaves the row ``queued`` for reaper rule c and still answers success.
    """
    async with _transaction(session, project):
        before = await _locked(session, job_id)
        if before.status not in (js.PAUSED, js.INTERRUPTED):
            raise rejection(
                409,
                'job_finished' if before.status in js.TERMINAL else 'resume_blocked',
                f'job {job_id} is {before.status}, not resumable',
            )
        blocker = await session.run_sync(lambda sync: resume_blocker(sync, before))
        if blocker is not None:
            raise rejection(409, 'resume_blocked', f'cannot resume: {blocker}', reason=blocker)
        try:
            moved = await session.run_sync(lambda sync: repo.resume(sync, job_id))
        except repo.JobBusyError as busy:
            raise rejection(
                409,
                'job_busy',
                'another job of this group is active',
                active_job_id=None if busy.active_job_id is None else str(busy.active_job_id),
            ) from None
        if moved is None:
            raise rejection(409, 'resume_blocked', f'job {job_id} is no longer resumable')
        await session.run_sync(lambda sync: job_events.emit_state(sync, moved))
    wake(project)
    await _dispatch_after_commit(session, project, moved, fail_on_broker_error=False)
    return ControlResult(moved, True)

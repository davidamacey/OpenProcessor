"""The task wrapper: claim, heartbeat, run the handler, write the terminal state (02b section 3.1).

``run_job`` is what every registered job task calls with its job id. In order: read the job's
project and kind; refuse a project that was archived or deleted while the job waited; claim the
row in one statement (a second delivery, a cancelled, paused or finished row, or a spent retry
budget claims nothing and is acknowledged); start the heartbeat thread; run the handler in an
event loop of its own; stop the heartbeat; write the one fenced terminal write. A zero-row fence
write means the lease was lost: nothing is written.

The handler runs as a task of a loop this module owns, with a cooperative soft-limit timer of its
own. Celery raises ``SoftTimeLimitExceeded`` from a signal handler in the main thread, which is
where the loop runs; spike S1 item 7 showed that with ``asyncio.to_thread`` the exception, and so
the terminal write, waited for the worker thread to end. The timer cancels the handler task at
the soft limit instead, and the loop is closed without joining executor threads (``asyncio.run``
would join them), so ``failed (time_limit)`` is written at once. Celery's own limits stay the
backstop. ``revoke`` is never used.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.db.engine import get_sync_engine, new_task_async_engine
from src.db.repos import job_events, jobs as repo
from src.domain import job_states as js, job_timing
from src.tasks.runtime import hooks
from src.tasks.runtime.context import JobContext, JobResult, Paused, Yield
from src.tasks.runtime.event_wakeup import wake
from src.tasks.runtime.heartbeat import JobHeartbeat
from src.tasks.runtime.scoped import head_transaction, job_transaction


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.config.job_kinds.kind import JobKind
    from src.tasks.runtime.context import ClientFactory

logger = logging.getLogger(__name__)

UNAVAILABLE_PROJECT_STATUSES = frozenset({'archived', 'deleting', 'failed'})
SOFT_LIMIT_LEAD_S = 2.0
_MESSAGE_CHARS = 2000


@dataclass(frozen=True)
class _Ran:
    """How the handler's run ended: ``return``, ``error``, ``cancelled``, ``time_limit``, ``exit``."""

    how: str
    value: Any = None
    error: BaseException | None = None


@dataclass(frozen=True)
class _Claim:
    job: repo.JobRow
    head: repo.JobHead


def owner_id() -> str:
    return f'{socket.gethostname()}:{os.getpid()}'


def run_job(
    kind: JobKind,
    handler: Callable[[JobContext], Awaitable[Any]],
    client_factories: Mapping[str, ClientFactory],
    job_id_text: str,
) -> None:
    """Run one delivery of a job. Returns when the message may be acknowledged."""
    job_id = uuid.UUID(job_id_text)
    claim = _claim(job_id)
    if claim is None:
        return
    beat = JobHeartbeat(job_id, claim.job.project_id, claim.job.fencing_token)
    beat.start()
    ran: _Ran | None = None
    try:
        ran = _execute(kind, handler, client_factories, claim, beat)
    finally:
        beat.stop()
        if ran is None:  # something outside the handler's own failures ended the delivery
            _write_interrupted(claim)
    _finalize(kind, claim, beat, ran)
    if ran.how == 'exit':
        raise ran.error  # type: ignore[misc]  # a SystemExit: Celery requeues the message


# --- claim ------------------------------------------------------------------------------------


def _claim(job_id: uuid.UUID) -> _Claim | None:
    with Session(get_sync_engine(), expire_on_commit=False) as session:
        with head_transaction(session):
            head = repo.read_head(session, job_id)
        if head is None:
            logger.warning('claim_skipped job_id=%s reason=missing', job_id)
            return None
        if head.project_status in UNAVAILABLE_PROJECT_STATUSES:
            _fail_unclaimed(
                session, head, 'project_unavailable', f'project is {head.project_status}'
            )
            return None
        with job_transaction(session, head.project_id):
            result = repo.claim(session, job_id, owner_id(), job_timing.LEASE_TTL_S)
            if isinstance(result, repo.Claimed):
                job_events.emit_state(session, result.job)
        if isinstance(result, repo.Claimed):
            _after_claim(head, result.job)
            return _Claim(result.job, head)
        _refused(session, head, result)
        return None


def _after_claim(head: repo.JobHead, job: repo.JobRow) -> None:
    structlog.contextvars.bind_contextvars(project=head.project_slug, attempt=job.attempt)
    queue = hooks.queue_hooks()
    if queue is not None:
        try:
            queue.clear_queued(str(job.id))
        except hooks.HookError as exc:
            logger.warning('queued_marker_not_cleared job_id=%s error=%s', job.id, exc)
    wake(job.project_id)


def _refused(session: Session, head: repo.JobHead, result: repo.ClaimRefused) -> None:
    if result.reason == 'exhausted':
        _fail_unclaimed(session, head, 'attempts_exhausted', 'every attempt ended in a failure')
        return
    logger.info(
        'claim_skipped job_id=%s reason=%s status=%s', head.id, result.reason, result.status
    )


def _fail_unclaimed(session: Session, head: repo.JobHead, code: str, message: str) -> None:
    """A queued job that will not run: ``failed``, with its event."""
    with job_transaction(session, head.project_id):
        failed = repo.fail_queued(session, head.id, code, message)
        if failed is not None:
            job_events.emit_state(session, failed)
    if failed is not None:
        wake(head.project_id)
        logger.warning('job_failed_unclaimed job_id=%s code=%s', head.id, code)


# --- the run ----------------------------------------------------------------------------------


def _execute(
    kind: JobKind,
    handler: Callable[[JobContext], Awaitable[Any]],
    client_factories: Mapping[str, ClientFactory],
    claim: _Claim,
    beat: JobHeartbeat,
) -> _Ran:
    """Run the handler on a loop this function owns and closes without joining threads."""
    server = hooks.choose_server(kind.name) if kind.gpu_affinity == 'any_card' else None
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_main(kind, handler, client_factories, claim, beat, server))
    except SoftTimeLimitExceeded:
        return _Ran('time_limit')
    except SystemExit as exc:
        return _Ran('exit', error=exc)
    finally:
        _cancel_leftovers(loop)
        loop.close()  # BaseEventLoop.close shuts the executor down without waiting


def _cancel_leftovers(loop: asyncio.AbstractEventLoop) -> None:
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except (SoftTimeLimitExceeded, SystemExit):
            logger.warning('leftover_tasks_not_awaited count=%s', len(pending))


async def _main(
    kind: JobKind,
    handler: Callable[[JobContext], Awaitable[Any]],
    client_factories: Mapping[str, ClientFactory],
    claim: _Claim,
    beat: JobHeartbeat,
    server: str | None,
) -> _Ran:
    engine = new_task_async_engine(f'op-job-{kind.name}')
    ctx = JobContext(
        claim.job,
        kind=kind,
        project_slug=claim.head.project_slug,
        beat=beat,
        engine=engine,
        client_factories=client_factories,
        server=server,
    )
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(handler(ctx))
    timed_out = False

    def soft_limit() -> None:
        nonlocal timed_out
        timed_out = True
        task.cancel()

    timer = loop.call_later(
        kind.soft_limit_s - min(SOFT_LIMIT_LEAD_S, kind.soft_limit_s * 0.1), soft_limit
    )
    try:
        await asyncio.wait({task})
        if task.cancelled():
            return _Ran('time_limit') if timed_out else _Ran('cancelled')
        error = task.exception()
        if isinstance(error, SoftTimeLimitExceeded):
            return _Ran('time_limit')
        if error is not None:
            logger.error(
                'job_handler_failed job_id=%s kind=%s', ctx.job_id, kind.name, exc_info=error
            )
            return _Ran('error', error=error)
        await ctx.flush_progress()
        return _Ran('return', value=task.result())
    finally:
        timer.cancel()
        await ctx.aclose()
        await engine.dispose()


# --- the terminal write -----------------------------------------------------------------------


def _write_interrupted(claim: _Claim) -> None:
    """Best effort when something outside the handler (a signal, a crash) ends the delivery."""
    try:
        with (
            Session(get_sync_engine(), expire_on_commit=False) as session,
            job_transaction(session, claim.job.project_id),
        ):
            moved = repo.interrupt(session, claim.job.id, claim.job.fencing_token)
            if moved is not None:
                job_events.emit_state(session, moved)
    except (
        SQLAlchemyError,
        OSError,
    ):  # the exception that ended the delivery is the one to surface
        logger.exception('interrupt_write_failed job_id=%s', claim.job.id)


def _finalize(kind: JobKind, claim: _Claim, beat: JobHeartbeat, ran: _Ran) -> None:
    job = claim.job
    with Session(get_sync_engine(), expire_on_commit=False) as session:
        with job_transaction(session, job.project_id):
            moved, dispatch = _write_terminal(session, kind, claim, beat, ran)
            if moved is not None:
                job_events.emit_state(session, moved)
        if moved is None:
            logger.warning('lease_lost job_id=%s token=%s', job.id, job.fencing_token)
            return
        wake(job.project_id)
        if dispatch and moved.status == js.QUEUED:
            _dispatch_again(session, kind, claim, moved)


def _write_terminal(
    session: Session, kind: JobKind, claim: _Claim, beat: JobHeartbeat, ran: _Ran
) -> tuple[repo.JobRow | None, bool]:
    """The fenced write for how the run ended. The second value says to dispatch afterwards."""
    job = claim.job
    token = job.fencing_token
    if ran.how == 'exit':
        return repo.interrupt(session, job.id, token), False
    if ran.how == 'time_limit':
        return _fail(
            session, job, 'time_limit', 'the soft time limit was reached', 'time_limit'
        ), False
    if ran.how == 'cancelled':
        return repo.finish(session, job.id, token, status=js.CANCELLED, outcome='cancelled'), False
    if ran.how == 'error':
        message = f'{type(ran.error).__name__}: {ran.error}'[:_MESSAGE_CHARS]
        return _fail(session, job, 'handler_error', message, 'failed'), False
    return _write_returned(session, kind, claim, beat, ran.value)


def _fail(
    session: Session, job: repo.JobRow, code: str, message: str, outcome: str
) -> repo.JobRow | None:
    return repo.finish(
        session,
        job.id,
        job.fencing_token,
        status=js.FAILED,
        outcome=outcome,
        error_code=code,
        error_message=message,
    )


def _write_returned(
    session: Session, kind: JobKind, claim: _Claim, beat: JobHeartbeat, value: Any
) -> tuple[repo.JobRow | None, bool]:
    job = claim.job
    token = job.fencing_token
    result, with_errors = _unwrap(value)
    if beat.cancel_requested.is_set():
        return (
            repo.finish(
                session,
                job.id,
                token,
                status=js.CANCELLED,
                outcome='cancelled',
                result=_mapping(result),
            ),
            False,
        )
    if isinstance(value, Paused) or (isinstance(value, Yield) and beat.pause_requested.is_set()):
        if kind.pausable:
            reason = value.reason if isinstance(value, Paused) else None
            return repo.pause_running(session, job.id, token, reason), False
        return _fail(session, job, 'job_not_pausable', f'{kind.name} cannot pause', 'failed'), False
    if isinstance(value, Yield):
        if kind.slice_s is None:
            return _fail(
                session, job, 'yield_not_allowed', f'{kind.name} has no slice_s', 'failed'
            ), False
        return repo.queue_yield(session, job.id, token), True
    try:
        stored = _mapping(result)
    except TypeError as exc:
        return _fail(session, job, 'result_invalid', str(exc)[:_MESSAGE_CHARS], 'failed'), False
    status = js.COMPLETED_WITH_ERRORS if with_errors else js.COMPLETED
    return repo.finish(session, job.id, token, status=status, outcome=status, result=stored), False


def _unwrap(value: Any) -> tuple[Any, bool]:
    if isinstance(value, JobResult):
        return value.result, value.with_errors
    return value, False


def _mapping(result: Any) -> Mapping[str, Any] | None:
    if result is None or isinstance(result, (Yield, Paused)):
        return None
    if isinstance(result, Mapping):
        return result
    raise TypeError(
        f'a handler returns a mapping, JobResult, Yield or Paused, not {type(result).__name__}'
    )


def _dispatch_again(session: Session, kind: JobKind, claim: _Claim, moved: repo.JobRow) -> None:
    """After a yield: send the message again and record it; rule c catches a failed send."""
    queue = hooks.queue_hooks()
    if queue is None:
        logger.error('yield_without_queue_hooks job_id=%s (reaper rule c will dispatch)', moved.id)
        return
    try:
        queue.enqueue(kind.task_name, job_id=str(moved.id))
        queue.mark_queued(str(moved.id), ttl_s=kind.max_queue_wait_s)
    except hooks.HookError as exc:  # the row stays queued for rule c
        logger.warning('yield_dispatch_failed job_id=%s error=%s', moved.id, exc)
        return
    with job_transaction(session, claim.job.project_id):
        repo.mark_dispatched(session, moved.id)

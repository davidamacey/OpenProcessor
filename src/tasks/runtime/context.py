"""What a job handler gets: ``JobContext``, and what it may return: ``Yield``, ``Paused``, ``JobResult``.

A handler is ``async def handler(ctx: JobContext)``. The wrapper (``job_task.py``) builds the
context inside the task's event loop, over a per-task ``NullPool`` async engine that it disposes
before the loop ends, and closes every client the handler asked for.

Cooperative control: ``cancelled()`` and ``paused()`` are plain reads of events the heartbeat
thread and every fenced write keep current; the handler checks them at a boundary (before each
unit of work, before a budget reservation) and returns. Nothing is interrupted from outside, and
Celery ``revoke`` is never used (D-71).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repos import job_events, jobs as repo
from src.domain import job_timing
from src.tasks.runtime.event_wakeup import wake
from src.tasks.runtime.scoped import async_job_transaction


if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator, Mapping

    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config.job_kinds.kind import JobKind
    from src.tasks.runtime.heartbeat import JobHeartbeat

ClientFactory = Callable[[], Awaitable[Any]]
"""Builds one client inside the task's loop; the wrapper closes it (``aclose`` or ``close``)."""


@dataclass(frozen=True)
class Yield:
    """Return at a boundary after the kind's ``slice_s``: the job goes back to ``queued``,
    behind other projects' jobs, and keeps its attempt budget (research G-2)."""


@dataclass(frozen=True)
class Paused:
    """Return after seeing ``ctx.paused()``: the job goes ``paused`` with the requested reason."""

    reason: str | None = None


@dataclass(frozen=True)
class JobResult:
    """A result that ended with errors (``completed_with_errors``); a plain mapping is ``completed``."""

    result: Mapping[str, Any] | None = None
    with_errors: bool = False


class JobContext:
    """The handler's view of one claimed job."""

    def __init__(
        self,
        job: repo.JobRow,
        *,
        kind: JobKind,
        project_slug: str | None,
        beat: JobHeartbeat,
        engine: AsyncEngine,
        client_factories: Mapping[str, ClientFactory] | None = None,
        server: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.job_id: uuid.UUID = job.id
        self.kind = kind.name
        self.project_id: uuid.UUID | None = job.project_id
        self.project_slug = project_slug
        self.params: Mapping[str, Any] = job.params
        self.resume_from: Any | None = job.checkpoint
        """The cursor the previous attempt last wrote with :meth:`checkpoint`, or ``None``."""
        self.attempt = job.attempt
        self.fencing_token = job.fencing_token
        self.server = server
        self._job = job
        self._spec = kind
        self._beat = beat
        self._engine = engine
        self._factories = dict(client_factories or {})
        self._clients: dict[str, Any] = {}
        self._clock = clock
        self._started = clock()
        self._last_progress_at: float | None = None
        self._last_stage: str | None = None
        self._unwritten: dict[str, Any] | None = None

    # --- cooperative control ---------------------------------------------------------------

    def cancelled(self) -> bool:
        """A cancel was requested, or the lease was lost (the job must stop either way)."""
        return self._beat.cancel_requested.is_set() or self._beat.lost.is_set()

    def paused(self) -> bool:
        """A pause was requested (by the route or by :meth:`request_pause`)."""
        return self._beat.pause_requested.is_set()

    def lost(self) -> bool:
        return self._beat.lost.is_set()

    def may_write(self) -> bool:
        """Whether a write outside the database is still safe (the lease is young enough)."""
        return self._beat.may_write()

    def slice_due(self) -> bool:
        """True once this delivery ran for the kind's ``slice_s``: return :class:`Yield`."""
        slice_s = self._spec.slice_s
        return slice_s is not None and self._clock() - self._started >= slice_s

    async def request_pause(self, reason: str) -> None:
        """Ask for this job's own pause (``audit``, ``budget``, ``project_paused``,
        ``vlm_unavailable``); then return :class:`Paused` at the next boundary."""
        async with AsyncSession(self._engine) as session, self._tx(session):
            stored = await session.run_sync(
                lambda s: repo.request_pause(s, self.job_id, self.fencing_token, reason)
            )
        if not stored:
            self._beat.lost.set()
            return
        self._beat.pause_requested.set()

    # --- fenced writes ---------------------------------------------------------------------

    async def progress(
        self,
        done: int,
        total: int | None = None,
        stage: str | None = None,
        counters: Mapping[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Record a full progress snapshot, at most once per ``PROGRESS_MIN_INTERVAL_S`` and
        always on a stage change; a skipped snapshot is kept and written with the next one."""
        snapshot = {'done': done, 'total': total, 'stage': stage, 'counters': dict(counters or {})}
        self._unwritten = snapshot
        now = self._clock()
        due = (
            force
            or self._last_progress_at is None
            or stage != self._last_stage
            or now - self._last_progress_at >= job_timing.PROGRESS_MIN_INTERVAL_S
        )
        if due:
            await self._write_progress(snapshot)

    async def flush_progress(self) -> None:
        """Write the last snapshot a throttled call skipped (the wrapper calls this at the end)."""
        if self._unwritten is not None:
            await self._write_progress(self._unwritten)

    async def _write_progress(self, snapshot: dict[str, Any]) -> None:
        async with AsyncSession(self._engine) as session, self._tx(session):
            control = await session.run_sync(
                lambda s: repo.progress(s, self.job_id, self.fencing_token, snapshot)
            )
            if control is not None:
                await session.run_sync(lambda s: job_events.emit_progress(s, self._job, snapshot))
        self._last_progress_at = self._clock()
        self._last_stage = snapshot['stage']
        self._unwritten = None
        if control is None:
            self._beat.lost.set()
            return
        self._beat.note_control(control.cancel_requested, control.pause_requested)
        await asyncio.to_thread(wake, self.project_id)

    async def checkpoint(self, cursor: Mapping[str, Any] | None) -> None:
        """Write the resume cursor, fenced; a resumable kind writes it after each durable chunk."""
        async with AsyncSession(self._engine) as session, self._tx(session):
            control = await session.run_sync(
                lambda s: repo.checkpoint(s, self.job_id, self.fencing_token, cursor)
            )
        if control is None:
            self._beat.lost.set()
            return
        self._beat.note_control(control.cancel_requested, control.pause_requested)

    # --- resources -------------------------------------------------------------------------

    def _tx(self, session: AsyncSession) -> Any:
        return async_job_transaction(session, self.project_id)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A session on the task's engine in the job's project scope; commits on a clean exit."""
        async with AsyncSession(self._engine, expire_on_commit=False) as session, self._tx(session):
            yield session

    async def client(self, name: str) -> Any:
        """The client the image registered under ``name``, built on first use inside the loop."""
        if name not in self._clients:
            try:
                factory = self._factories[name]
            except KeyError:
                raise KeyError(
                    f'no client factory {name!r} was registered for {self.kind}'
                ) from None
            self._clients[name] = await factory()
        return self._clients[name]

    async def aclose(self) -> None:
        """Close every client the handler built (the wrapper calls this before the loop ends)."""
        for client in self._clients.values():
            closer = getattr(client, 'aclose', None) or getattr(client, 'close', None)
            if closer is None:
                continue
            outcome = closer()
            if inspect.isawaitable(outcome):
                await outcome
        self._clients.clear()

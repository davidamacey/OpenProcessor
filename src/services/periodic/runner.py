"""The frame every periodic task body runs in (02b section 4.1).

A tick is a plain Celery task that starts a short asyncio program on the child's own event loop
(``run_with_engine``: a one-connection ``NullPool`` engine bound for the program, disposed
after it). Bodies that must not overlap themselves take a transaction-level advisory lock,
``pg_try_advisory_xact_lock(class, obj)``: false means another tick holds it and this one skips.
The lock transaction stays open while the body runs and ends (releasing the lock) when the body
returns, and it works behind a transaction pooler
(https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS).

A per-project body runs once per active, unpaused project with that project bound. One project's
failure does not stop the others; the tick raises :class:`PeriodicTickError` at the end, so the
failure is on the task, not only in a log line.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.config.project_context import bind_project
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.engine import current_async_engine, get_sync_engine, run_with_engine
from src.db.repos import projects as projects_repo
from src.db.session import sync_deployment_tx
from src.services.projects.records import record_from_row


if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator, Awaitable, Callable

    from src.config.projects import ProjectRecord

logger = get_logger(__name__)

_LOCK = text('SELECT pg_try_advisory_xact_lock(:class, :obj)')


@dataclass(frozen=True)
class TickProject:
    """One project a tick acts on: its id, its advisory-lock key and the record to bind."""

    id: uuid.UUID
    slug: str
    lock_key: int
    record: ProjectRecord


@dataclass
class TickReport:
    done: list[str] = field(default_factory=list)
    skipped_busy: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class PeriodicTickError(RuntimeError):
    """A tick ran every project it could and at least one failed."""


def read_tick_projects() -> list[TickProject]:
    """Every ``active`` project that is not paused, sorted by slug (sync; call off the loop)."""
    with (
        Session(get_sync_engine()) as session,
        sync_deployment_tx(session, 'periodic tick projects'),
    ):
        rows = projects_repo.list_projects_sync(session)
        return [
            TickProject(r.id, r.slug, r.lock_key, record_from_row(r))
            for r in sorted(rows, key=lambda r: r.slug)
            if r.status == 'active' and r.paused_at is None
        ]


@asynccontextmanager
async def tick_lock(lock_class: int, obj: int = 0) -> AsyncIterator[bool]:
    """``True`` while this tick holds ``(lock_class, obj)``; ``False`` when another tick does."""
    async with current_async_engine().connect() as conn, conn.begin():
        got = (await conn.execute(_LOCK, {'class': lock_class, 'obj': obj})).scalar_one()
        yield bool(got)


async def for_each_project(
    lock_class: int,
    body: Callable[[TickProject], Awaitable[object]],
    *,
    task: str,
) -> TickReport:
    """Run ``body`` for each tickable project under its advisory lock, the project bound."""
    report = TickReport()
    for project in await asyncio.to_thread(read_tick_projects):
        try:
            async with tick_lock(lock_class, project.lock_key) as held:
                if not held:
                    report.skipped_busy.append(project.slug)
                    continue
                with bind_project(project.record):
                    await body(project)
            report.done.append(project.slug)
        except (SQLAlchemyError, RequestRejectedError, OSError) as exc:
            logger.warning(
                'periodic_project_failed', task=task, project=project.slug, error=str(exc)
            )
            report.failed[project.slug] = f'{type(exc).__name__}: {exc}'
    if report.failed:
        raise PeriodicTickError(f'{task}: {report.failed}')
    return report


def run_periodic[T](name: str, body: Awaitable[T]) -> T:
    """Run one tick's coroutine on a new event loop with its own engine bound."""
    return asyncio.run(run_with_engine(body, f'op-periodic-{name}', one_shot=True))


__all__ = [
    'PeriodicTickError',
    'TickProject',
    'TickReport',
    'for_each_project',
    'read_tick_projects',
    'run_periodic',
    'tick_lock',
]

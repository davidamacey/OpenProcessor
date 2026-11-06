"""The ``events`` table: the durable UI feed and the SSE resume cursor.

Writers (the job runtime, lifecycle routes) insert in the caller's transaction so a rolled-back
change emits nothing. Readers resume from a ``(xid, id)`` cursor and see only rows whose
transaction is older than every running one, because an identity column is assigned at insert
and two concurrent writers can commit out of id order (02b section 6.3,
https://www.postgresql.org/docs/current/functions-info.html#FUNCTIONS-PG-SNAPSHOT).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, insert, select, text

from src.db.models.jobs import EVENT_PAYLOAD_MAX_BYTES, Event
from src.db.models.projects import Project


if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import timedelta

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session

PAGE_SIZE = 500
DELETE_BATCH = 10_000


class EventPayloadTooLargeError(ValueError):
    """The payload would break the table's size ``CHECK`` (7,000 bytes as ``jsonb`` text)."""


class UnknownProjectError(LookupError):
    """An event named a project slug that is not in ``projects``."""


@dataclass(frozen=True)
class InsertedEvent:
    id: int
    project_id: uuid.UUID | None


@dataclass(frozen=True)
class EventRow:
    xid: int
    id: int
    type: str
    payload: dict[str, Any]


def _check_size(payload: Mapping[str, Any]) -> None:
    # ``jsonb`` text puts a space after ``:`` and ``,``; measuring that form keeps the check honest.
    size = len(json.dumps(payload, separators=(', ', ': '), default=str).encode())
    if size > EVENT_PAYLOAD_MAX_BYTES:
        msg = f'event payload is {size} bytes; the limit is {EVENT_PAYLOAD_MAX_BYTES}'
        raise EventPayloadTooLargeError(msg)


def _resolve_project(session: Session, project: uuid.UUID | str | None) -> uuid.UUID | None:
    """A project id, or a slug resolved through ``projects``; ``None`` is the global stream."""
    if project is None:
        return None
    if isinstance(project, uuid.UUID):
        return project
    try:
        return uuid.UUID(project)
    except ValueError:
        pass
    found = session.execute(select(Project.id).where(Project.slug == project)).scalar_one_or_none()
    if found is None:
        raise UnknownProjectError(project)
    return found


def insert_event(
    session: Session,
    project: uuid.UUID | str | None,
    type: str,  # noqa: A002  # the column's name; the signature is fixed by the job-runtime lane
    payload: Mapping[str, Any],
    *,
    job_id: uuid.UUID | None = None,
) -> InsertedEvent:
    """Insert one event in the caller's transaction (no commit here).

    ``project`` is a project id, a slug, or ``None`` for the global stream (a core or deployment
    scoped transaction; row-level security refuses it elsewhere).
    """
    _check_size(payload)
    project_id = _resolve_project(session, project)
    new_id = session.execute(
        insert(Event)
        .values(project_id=project_id, job_id=job_id, type=type, payload=dict(payload))
        .returning(Event.id)
    ).scalar_one()
    return InsertedEvent(id=new_id, project_id=project_id)


async def insert_event_async(
    session: AsyncSession,
    project: uuid.UUID | str | None,
    type: str,  # noqa: A002  # the signature is fixed by the job-runtime lane
    payload: Mapping[str, Any],
    *,
    job_id: uuid.UUID | None = None,
) -> InsertedEvent:
    """:func:`insert_event` for an ``AsyncSession``."""
    return await session.run_sync(
        lambda sync: insert_event(sync, project, type, payload, job_id=job_id)
    )


# Static statements: the project and global forms differ only in the project filter, and each
# is kept whole so the planner-facing text is greppable (and the EXPLAIN test can run it).
_PAGE_PROJECT = (
    'SELECT xid::text AS xid, id, type, payload FROM events '
    'WHERE project_id = :project_id AND (xid, id) > (CAST(:xid AS xid8), :id) '
    'AND xid < pg_snapshot_xmin(pg_current_snapshot()) ORDER BY xid, id LIMIT :limit'
)
_PAGE_GLOBAL = (
    'SELECT xid::text AS xid, id, type, payload FROM events '
    'WHERE project_id IS NULL AND (xid, id) > (CAST(:xid AS xid8), :id) '
    'AND xid < pg_snapshot_xmin(pg_current_snapshot()) ORDER BY xid, id LIMIT :limit'
)
_HEAD_PROJECT = (
    'SELECT xid::text AS xid, id FROM events WHERE project_id = :project_id '
    'AND xid < pg_snapshot_xmin(pg_current_snapshot()) ORDER BY xid DESC, id DESC LIMIT 1'
)
_HEAD_GLOBAL = (
    'SELECT xid::text AS xid, id FROM events WHERE project_id IS NULL '
    'AND xid < pg_snapshot_xmin(pg_current_snapshot()) ORDER BY xid DESC, id DESC LIMIT 1'
)
_OLDEST_PROJECT = 'SELECT xid::text AS xid, id FROM events WHERE project_id = :project_id ORDER BY xid, id LIMIT 1'
_OLDEST_GLOBAL = (
    'SELECT xid::text AS xid, id FROM events WHERE project_id IS NULL ORDER BY xid, id LIMIT 1'
)


def _page_sql(project_id: uuid.UUID | None) -> str:
    return _PAGE_GLOBAL if project_id is None else _PAGE_PROJECT


def _bind(project_id: uuid.UUID | None) -> dict[str, Any]:
    return {} if project_id is None else {'project_id': project_id}


async def fetch_after(
    session: AsyncSession,
    project_id: uuid.UUID | None,
    xid: int,
    last_id: int,
    *,
    limit: int = PAGE_SIZE,
) -> list[EventRow]:
    """Committed-safe rows after the cursor, in cursor order (02b section 6.3)."""
    params: dict[str, Any] = {'xid': str(xid), 'id': last_id, 'limit': limit, **_bind(project_id)}
    result = await session.execute(text(_page_sql(project_id)), params)
    return [EventRow(int(r.xid), r.id, r.type, r.payload) for r in result]


async def head_cursor(session: AsyncSession, project_id: uuid.UUID | None) -> tuple[int, int]:
    """The newest visible ``(xid, id)`` for a stream, ``(0, 0)`` when it has no rows."""
    sql = _HEAD_GLOBAL if project_id is None else _HEAD_PROJECT
    row = (await session.execute(text(sql), _bind(project_id))).first()
    return (0, 0) if row is None else (int(row.xid), row.id)


async def oldest_cursor(
    session: AsyncSession, project_id: uuid.UUID | None
) -> tuple[int, int] | None:
    """The oldest retained ``(xid, id)`` of a stream, ``None`` when it has no rows."""
    sql = _OLDEST_GLOBAL if project_id is None else _OLDEST_PROJECT
    row = (await session.execute(text(sql), _bind(project_id))).first()
    return None if row is None else (int(row.xid), row.id)


async def cursor_from_future(session: AsyncSession, xid: int) -> bool:
    """True when ``xid`` is beyond this cluster's transaction counter (a cursor from before a restore)."""
    row = (
        await session.execute(
            text('SELECT CAST(:xid AS xid8) >= pg_snapshot_xmax(pg_current_snapshot()) AS future'),
            {'xid': str(xid)},
        )
    ).one()
    return bool(row.future)


async def project_id_for_slug(session: AsyncSession, slug: str) -> uuid.UUID | None:
    return (
        await session.execute(select(Project.id).where(Project.slug == slug))
    ).scalar_one_or_none()


def delete_older_than(session: Session, age: timedelta, *, batch: int = DELETE_BATCH) -> int:
    """Delete up to ``batch`` events older than ``age``; returns the count (retention, reaper rule f).

    One bounded batch per call (trap T-14: no unbounded delete); the caller loops while the
    count equals ``batch``. Needs a deployment-scoped transaction.
    """
    oldest = (
        select(Event.id).where(Event.created_at < func.now() - age).order_by(Event.id).limit(batch)
    )
    return len(session.execute(delete(Event).where(Event.id.in_(oldest)).returning(Event.id)).all())

"""Ingest sources and the files each one has seen: the queue a bulk run drains.

``ingest_sources`` holds the definition (kind, config, triggers) and the per-source scan and
trigger state; ``ingest_source_files`` holds one row per discovered file. A file is ``pending``
until a run claims it, ``claimed`` while a run holds it, then ``done`` or ``failed``. A failed
row is retried until it has been tried ``MAX_ATTEMPTS`` times; a row that gave up (or failed for
a reason a retry cannot fix) is *dead* and waits for ``retry_dead``.

Every function takes the caller's sync ``Session`` inside the caller's scoped transaction and
never commits; an async caller wraps one in ``AsyncSession.run_sync``.

The claim is ``FOR UPDATE SKIP LOCKED``: two runners reading the same source take disjoint
batches and never wait for each other
(https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE). An insert that
meets a known file does nothing (``ON CONFLICT DO NOTHING``), which is the idempotency of the
whole feature: a replayed event, a repeated scan and an overlapping pass queue nothing twice
(https://www.postgresql.org/docs/current/sql-insert.html#SQL-ON-CONFLICT).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from src.db.models.datasets import IngestSource, IngestSourceFile


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Mapping, Sequence

    from sqlalchemy.orm import Session

MAX_ATTEMPTS = 3
PENDING = 'pending'
CLAIMED = 'claimed'
DONE = 'done'
FAILED = 'failed'
# Rows per INSERT: three bound values each, well under the driver's parameter limit.
INSERT_CHUNK = 1000
_NAME_UNIQUE = 'uq_ingest_sources_project_id_name'


class SourceNameTakenError(Exception):
    """The project already has a source of this name."""


@dataclass(frozen=True)
class SourceRow:
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    kind: str
    config: dict[str, Any]
    schedule: dict[str, Any]
    enabled: bool
    revision: int
    last_poll_at: dt.datetime | None
    scan_state: dict[str, Any]
    created_at: dt.datetime
    updated_at: dt.datetime


@dataclass(frozen=True)
class ClaimedFile:
    file_key: str
    attempts: int


def _row(model: IngestSource) -> SourceRow:
    return SourceRow(
        id=model.id,
        project_id=model.project_id,
        name=model.name,
        kind=model.kind,
        config=dict(model.config),
        schedule=dict(model.schedule),
        enabled=model.enabled,
        revision=model.revision,
        last_poll_at=model.last_poll_at,
        scan_state=dict(model.scan_state or {}),
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


# --- definitions ------------------------------------------------------------------------------


def get(session: Session, source_id: uuid.UUID) -> SourceRow | None:
    found = session.get(IngestSource, source_id)
    return None if found is None else _row(found)


def get_by_name(session: Session, project_id: uuid.UUID, name: str) -> SourceRow | None:
    found = session.execute(
        select(IngestSource).where(IngestSource.project_id == project_id, IngestSource.name == name)
    ).scalar_one_or_none()
    return None if found is None else _row(found)


def list_for_project(session: Session, project_id: uuid.UUID) -> list[SourceRow]:
    found = session.execute(
        select(IngestSource)
        .where(IngestSource.project_id == project_id)
        .order_by(IngestSource.created_at, IngestSource.id)
    ).scalars()
    return [_row(m) for m in found]


def insert(
    session: Session,
    project_id: uuid.UUID,
    *,
    name: str,
    kind: str,
    config: Mapping[str, Any],
    schedule: Mapping[str, Any],
    enabled: bool = True,
) -> SourceRow:
    model = IngestSource(
        project_id=project_id,
        name=name,
        kind=kind,
        config=dict(config),
        schedule=dict(schedule),
        enabled=enabled,
        scan_state={},
    )
    try:
        with session.begin_nested():
            session.add(model)
            session.flush()
    except IntegrityError as exc:
        if _NAME_UNIQUE in str(exc.orig):
            raise SourceNameTakenError(name) from None
        raise
    session.refresh(model)
    return _row(model)


def update_definition(
    session: Session,
    source_id: uuid.UUID,
    *,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
    enabled: bool | None = None,
) -> SourceRow | None:
    """Replace the named parts of a definition; the revision advances by one."""
    given = {
        'name': name,
        'config': None if config is None else dict(config),
        'schedule': None if schedule is None else dict(schedule),
        'enabled': enabled,
    }
    values: dict[str, Any] = {
        'revision': IngestSource.revision + 1,
        'updated_at': func.now(),
        **{k: v for k, v in given.items() if v is not None},
    }
    try:
        with session.begin_nested():
            changed = session.execute(
                update(IngestSource)
                .where(IngestSource.id == source_id)
                .values(**values)
                .returning(IngestSource.id)
            ).scalar_one_or_none()
    except IntegrityError as exc:
        if _NAME_UNIQUE in str(exc.orig):
            raise SourceNameTakenError(str(name)) from None
        raise
    return None if changed is None else get(session, source_id)


def delete_source(session: Session, source_id: uuid.UUID) -> bool:
    """Remove the definition; its files go with it (``ON DELETE CASCADE``)."""
    return (
        session.execute(
            delete(IngestSource).where(IngestSource.id == source_id).returning(IngestSource.id)
        ).scalar_one_or_none()
        is not None
    )


# --- scan and trigger state -------------------------------------------------------------------


def state_get(session: Session, source_id: uuid.UUID, key: str, default: Any = None) -> Any:
    state = session.execute(
        select(IngestSource.scan_state).where(IngestSource.id == source_id)
    ).scalar_one_or_none()
    return default if not state or key not in state else state[key]


_SET_KEY = text(
    "UPDATE ingest_sources SET scan_state = jsonb_set(coalesce(scan_state, '{}'::jsonb), "
    'ARRAY[:key], CAST(:value AS jsonb)), updated_at = now() WHERE id = :id'
)
_MERGE = text(
    "UPDATE ingest_sources SET scan_state = coalesce(scan_state, '{}'::jsonb) || "
    'CAST(:value AS jsonb), updated_at = now() WHERE id = :id'
)


def state_set(session: Session, source_id: uuid.UUID, key: str, value: Any) -> None:
    """Set one key of the state; the other keys are untouched (a scan thread and a webhook do not
    overwrite each other)."""
    session.execute(_SET_KEY, {'id': source_id, 'key': key, 'value': json.dumps(value)})


def state_update(session: Session, source_id: uuid.UUID, values: Mapping[str, Any]) -> None:
    """Merge ``values`` into the state; a ``None`` value is stored as JSON null."""
    session.execute(_MERGE, {'id': source_id, 'value': json.dumps(dict(values))})


def state_add(session: Session, source_id: uuid.UUID, amounts: Mapping[str, int]) -> None:
    """Add to several counters of the state in one statement each (webhook totals)."""
    session.execute(
        select(IngestSource.id).where(IngestSource.id == source_id).with_for_update()
    ).scalar_one_or_none()
    for key, amount in amounts.items():
        current = state_get(session, source_id, key, 0)
        state_set(session, source_id, key, int(current or 0) + amount)


# --- files ------------------------------------------------------------------------------------


def enqueue(session: Session, source_id: uuid.UUID, file_keys: Sequence[str]) -> tuple[int, int]:
    """Insert the keys not yet known as ``pending``; returns ``(queued, duplicates)``."""
    keys = list(dict.fromkeys(file_keys))
    queued = 0
    for start in range(0, len(keys), INSERT_CHUNK):
        chunk = keys[start : start + INSERT_CHUNK]
        inserted = session.execute(
            pg_insert(IngestSourceFile)
            .values([{'source_id': source_id, 'file_key': k, 'status': PENDING} for k in chunk])
            .on_conflict_do_nothing(index_elements=['source_id', 'file_key'])
            .returning(IngestSourceFile.file_key)
        )
        queued += len(inserted.all())
    return queued, len(file_keys) - queued


_CLAIM = text(
    """
    WITH picked AS (
      SELECT file_key FROM ingest_source_files
      WHERE source_id = :source AND (status = 'pending' OR (status = 'failed' AND attempts < :max))
      ORDER BY enqueued_at, file_key
      LIMIT :n
      FOR UPDATE SKIP LOCKED
    )
    UPDATE ingest_source_files f
    SET status = 'claimed', job_id = :job, updated_at = now()
    FROM picked
    WHERE f.source_id = :source AND f.file_key = picked.file_key
    RETURNING f.file_key, f.attempts
    """
)


def claim(
    session: Session, source_id: uuid.UUID, job_id: uuid.UUID, limit: int
) -> list[ClaimedFile]:
    """Take up to ``limit`` files for the run ``job_id``, oldest first, skipping rows another
    runner holds locked."""
    rows = session.execute(
        _CLAIM, {'source': source_id, 'job': job_id, 'n': limit, 'max': MAX_ATTEMPTS}
    )
    return sorted((ClaimedFile(r.file_key, r.attempts) for r in rows), key=lambda c: c.file_key)


def release_claims(session: Session, source_id: uuid.UUID, job_id: uuid.UUID | None = None) -> int:
    """Put ``claimed`` rows of the source back to ``pending``: those of run ``job_id``, or, with
    no job, every one (a run starts only when none is live, so none is held by a live one)."""
    where = [IngestSourceFile.source_id == source_id, IngestSourceFile.status == CLAIMED]
    if job_id is not None:
        where.append(IngestSourceFile.job_id == job_id)
    return int(
        session.execute(
            update(IngestSourceFile).where(*where).values(status=PENDING, updated_at=func.now())
        ).rowcount  # type: ignore[attr-defined]
    )


_MARK_DONE = text(
    """
    UPDATE ingest_source_files f
    SET status = 'done', job_id = :job, image_id = v.image_id, error = NULL, updated_at = now()
    FROM (SELECT unnest(CAST(:keys AS text[])) AS file_key,
                 unnest(CAST(:images AS text[])) AS image_id) v
    WHERE f.source_id = :source AND f.file_key = v.file_key
    """
)
_MARK_FAILED = text(
    """
    UPDATE ingest_source_files f
    SET status = 'failed', job_id = :job, error = v.error, updated_at = now(),
        attempts = CASE WHEN v.terminal THEN :max ELSE f.attempts + 1 END
    FROM (SELECT unnest(CAST(:keys AS text[])) AS file_key,
                 unnest(CAST(:errors AS text[])) AS error,
                 unnest(CAST(:terminals AS boolean[])) AS terminal) v
    WHERE f.source_id = :source AND f.file_key = v.file_key AND f.status <> 'done'
    """
)


def mark_done(
    session: Session,
    source_id: uuid.UUID,
    job_id: uuid.UUID,
    rows: Sequence[tuple[str, str | None]],
) -> None:
    """Settle ``(file_key, image_id)`` rows as ``done``."""
    if rows:
        session.execute(
            _MARK_DONE,
            {
                'source': source_id,
                'job': job_id,
                'keys': [k for k, _ in rows],
                'images': [i for _, i in rows],
            },
        )


def error_text(message: str, kind: str) -> str:
    """The stored ``error``: the reason, led by the failure kind when there is one."""
    return f'[{kind}] {message}'[:300] if kind else message[:300]


def split_error(stored: str) -> tuple[str, str]:
    """``(error_kind, message)`` back out of a stored ``error``."""
    if stored.startswith('[') and '] ' in stored:
        kind, _, message = stored[1:].partition('] ')
        return kind, message
    return '', stored


def mark_failed(
    session: Session, source_id: uuid.UUID, job_id: uuid.UUID, rows: Sequence[tuple[str, str, bool]]
) -> None:
    """Record ``(file_key, error, terminal)``: the attempt counts, and a terminal failure (one a
    retry cannot fix) uses up the attempts so the row is dead at once."""
    if rows:
        session.execute(
            _MARK_FAILED,
            {
                'source': source_id,
                'job': job_id,
                'max': MAX_ATTEMPTS,
                'keys': [k for k, _, _ in rows],
                'errors': [e for _, e, _ in rows],
                'terminals': [t for _, _, t in rows],
            },
        )


def retry_dead(session: Session, source_id: uuid.UUID) -> int:
    """Failed rows out of attempts go back to ``pending`` with a clean count."""
    return int(
        session.execute(
            update(IngestSourceFile)
            .where(
                IngestSourceFile.source_id == source_id,
                IngestSourceFile.status == FAILED,
                IngestSourceFile.attempts >= MAX_ATTEMPTS,
            )
            .values(status=PENDING, attempts=0, updated_at=func.now())
        ).rowcount  # type: ignore[attr-defined]
    )


def requeue_all(session: Session, source_id: uuid.UUID) -> int:
    """Every ``done`` or ``failed`` row back to ``pending``: a fresh pass over a one-shot source."""
    return int(
        session.execute(
            update(IngestSourceFile)
            .where(
                IngestSourceFile.source_id == source_id,
                IngestSourceFile.status.in_((DONE, FAILED)),
            )
            .values(status=PENDING, attempts=0, error=None, updated_at=func.now())
        ).rowcount  # type: ignore[attr-defined]
    )


_COUNTS = text(
    'SELECT status, attempts >= :max AS dead, count(*) FROM ingest_source_files '
    'WHERE source_id = :source GROUP BY 1, 2'
)


def counts(session: Session, source_id: uuid.UUID) -> dict[str, int]:
    """Rows per state: ``pending``, ``claimed``, ``done``, ``failed`` (retried by a later run) and
    ``dead`` (out of attempts)."""
    found = {
        (status, dead): n
        for status, dead, n in session.execute(_COUNTS, {'source': source_id, 'max': MAX_ATTEMPTS})
    }
    failed_dead = found.get((FAILED, True), 0)
    return {
        'pending': found.get((PENDING, False), 0) + found.get((PENDING, True), 0),
        'claimed': found.get((CLAIMED, False), 0) + found.get((CLAIMED, True), 0),
        'done': found.get((DONE, False), 0) + found.get((DONE, True), 0),
        'failed': found.get((FAILED, False), 0),
        'dead': failed_dead,
    }


def errors(session: Session, source_id: uuid.UUID, limit: int = 50) -> list[dict[str, Any]]:
    """The most recent failed rows, for the status view."""
    rows = session.execute(
        select(IngestSourceFile)
        .where(IngestSourceFile.source_id == source_id, IngestSourceFile.status == FAILED)
        .order_by(IngestSourceFile.updated_at.desc(), IngestSourceFile.file_key)
        .limit(limit)
    ).scalars()
    return [
        {
            'key': r.file_key,
            'state': 'dead' if r.attempts >= MAX_ATTEMPTS else 'failed',
            'attempts': r.attempts,
            'error': split_error(r.error or '')[1],
        }
        for r in rows
    ]


def failures(
    session: Session, source_id: uuid.UUID, job_id: uuid.UUID, *, offset: int, limit: int
) -> tuple[list[dict[str, str]], int]:
    """``(page, total)`` of the files the run ``job_id`` left ``failed``, as ``path``, ``error``
    and ``error_kind``."""
    where = (
        IngestSourceFile.source_id == source_id,
        IngestSourceFile.job_id == job_id,
        IngestSourceFile.status == FAILED,
    )
    total = session.execute(select(func.count()).select_from(IngestSourceFile).where(*where))
    page = session.execute(
        select(IngestSourceFile)
        .where(*where)
        .order_by(IngestSourceFile.file_key)
        .offset(offset)
        .limit(limit)
    ).scalars()
    rows = []
    for r in page:
        kind, message = split_error(r.error or '')
        rows.append({'path': r.file_key, 'error': message, 'error_kind': kind})
    return rows, int(total.scalar_one())

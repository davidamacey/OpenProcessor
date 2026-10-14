"""The durable per-source queue: what a source has seen, and what is left to ingest.

One SQLite file per source (WAL, on the shared jobs volume). Every trigger of a
source (a watcher scan, a webhook event, a cron fire, a manual run) does the same
thing here: :meth:`ObjectQueue.enqueue` rows keyed by ``(key, etag)``. A bulk run of
kind ``queue`` drains the ``pending`` rows. The primary key is the idempotency of
the whole feature: a replayed event, a repeated scan and an overlapping pass insert
nothing.

Row states: ``pending`` (to ingest), ``done``, ``failed`` (retried by a later run
until :data:`MAX_ATTEMPTS`), ``dead`` (will not be retried until
:meth:`retry_dead`). ``meta`` holds the scan checkpoint and counters as JSON.

For an S3 source ``key`` is the object key and ``etag`` its ETag; for a folder
source ``key`` is the path and ``etag`` is ``"<mtime_ns>:<size>"``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

MAX_ATTEMPTS = 3
STATE_PENDING = 'pending'
STATE_DONE = 'done'
STATE_FAILED = 'failed'
STATE_DEAD = 'dead'
_FAIL_SQL = (
    'UPDATE objects SET attempts=attempts+1, error=?, updated_at=?,'
    ' state=CASE WHEN ? OR attempts+1>=? THEN ? ELSE ? END WHERE '
)
_FAIL_SQL_ONE_ETAG = _FAIL_SQL + 'key=? AND etag=? AND state!=?'
_FAIL_SQL_ANY_ETAG = _FAIL_SQL + 'key=? AND state!=?'
STATES = (STATE_PENDING, STATE_DONE, STATE_FAILED, STATE_DEAD)
# Failures that retrying cannot fix.
TERMINAL_ERROR_KINDS = frozenset(
    {'gone', 'too_large', 'unsupported_type', 'empty', 'decode_failed', 'unservable_path'}
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    key TEXT NOT NULL,
    etag TEXT NOT NULL,
    version TEXT,
    size INTEGER,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    queued_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (key, etag)
);
CREATE INDEX IF NOT EXISTS objects_state ON objects (state);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


@dataclass(frozen=True)
class QueueEntry:
    key: str
    etag: str
    version: str | None = None
    size: int | None = None
    attempts: int = 0


class ObjectQueue:
    """Opens a short-lived connection per call: safe from any thread or worker process."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15.0, isolation_level=None)
        try:
            conn.execute('PRAGMA busy_timeout=15000')
            conn.execute('PRAGMA synchronous=NORMAL')
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ writes

    def enqueue(self, entries: Iterable[QueueEntry]) -> tuple[int, int]:
        """Insert ``entries`` that are not yet known; returns ``(queued, duplicates)``.
        One transaction, so a webhook batch is all-or-nothing and fast."""
        now = time.time()
        rows = [(e.key, e.etag, e.version, e.size, STATE_PENDING, now, now) for e in entries]
        if not rows:
            return 0, 0
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            before = conn.total_changes
            conn.executemany(
                'INSERT OR IGNORE INTO objects (key, etag, version, size, state, queued_at,'
                ' updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
                rows,
            )
            queued = conn.total_changes - before
            conn.execute('COMMIT')
        return queued, len(rows) - queued

    def mark_done(self, items: Iterable[tuple[str, str | None]]) -> None:
        """Mark ``(key, etag)`` done; an ``etag`` of ``None`` marks every pending row of the key."""
        now = time.time()
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            for key, etag in items:
                if etag is None:
                    conn.execute(
                        'UPDATE objects SET state=?, error=NULL, updated_at=? WHERE key=? AND state!=?',
                        (STATE_DONE, now, key, STATE_DONE),
                    )
                else:
                    conn.execute(
                        'UPDATE objects SET state=?, error=NULL, updated_at=? WHERE key=? AND etag=?',
                        (STATE_DONE, now, key, etag),
                    )
            conn.execute('COMMIT')

    def mark_failed(self, items: Iterable[tuple[str, str | None, str, str]]) -> None:
        """Record failures as ``(key, etag, error, error_kind)``. A terminal kind, or the
        third attempt, makes the row ``dead``; otherwise it stays retryable."""
        now = time.time()
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            for key, etag, error, kind in items:
                terminal = kind in TERMINAL_ERROR_KINDS
                conn.execute(
                    _FAIL_SQL_ANY_ETAG if etag is None else _FAIL_SQL_ONE_ETAG,
                    (
                        error[:300],
                        now,
                        int(terminal),
                        MAX_ATTEMPTS,
                        STATE_DEAD,
                        STATE_FAILED,
                        *((key,) if etag is None else (key, etag)),
                        STATE_DONE,
                    ),
                )
            conn.execute('COMMIT')

    def retry_dead(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                'UPDATE objects SET state=?, attempts=0, updated_at=? WHERE state=?',
                (STATE_PENDING, time.time(), STATE_DEAD),
            )
            return cur.rowcount

    # ------------------------------------------------------------------- reads

    def iter_pending(self, *, page: int = 500) -> Iterator[QueueEntry]:
        """Entries still to ingest (``pending``, and ``failed`` with attempts left), oldest
        first, read a page at a time so a large backlog is never held in memory."""
        last = 0
        while True:
            with self._connect() as conn:
                rows = conn.execute(
                    'SELECT rowid, key, etag, version, size, attempts FROM objects'
                    ' WHERE rowid>? AND state IN (?, ?) ORDER BY rowid LIMIT ?',
                    (last, STATE_PENDING, STATE_FAILED, page),
                ).fetchall()
            if not rows:
                return
            for rowid, key, etag, version, size, attempts in rows:
                last = rowid
                yield QueueEntry(key, etag, version, size, attempts)

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            found = dict(conn.execute('SELECT state, COUNT(*) FROM objects GROUP BY state'))
        return {state: int(found.get(state, 0)) for state in STATES}

    def errors(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """The most recent failed or dead rows, for the status view."""
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT key, etag, state, attempts, error FROM objects WHERE state IN (?, ?)'
                ' ORDER BY updated_at DESC LIMIT ?',
                (STATE_FAILED, STATE_DEAD, limit),
            ).fetchall()
        return [
            {'key': k, 'etag': e, 'state': s, 'attempts': a, 'error': err or ''}
            for k, e, s, a, err in rows
        ]

    def known(self, key: str, etag: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute('SELECT 1 FROM objects WHERE key=? AND etag=?', (key, etag)).fetchone()
                is not None
            )

    # -------------------------------------------------------------------- meta

    def meta_get(self, name: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute('SELECT v FROM meta WHERE k=?', (name,)).fetchone()
        return json.loads(row[0]) if row else default

    def meta_set(self, name: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v',
                (name, json.dumps(value)),
            )

    def add_counters(self, amounts: dict[str, int]) -> None:
        """Add to several counters in one transaction (webhook totals)."""
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            for name, amount in amounts.items():
                row = conn.execute('SELECT v FROM meta WHERE k=?', (name,)).fetchone()
                total = (json.loads(row[0]) if row else 0) + amount
                conn.execute(
                    'INSERT INTO meta (k, v) VALUES (?, ?)'
                    ' ON CONFLICT(k) DO UPDATE SET v=excluded.v',
                    (name, json.dumps(total)),
                )
            conn.execute('COMMIT')

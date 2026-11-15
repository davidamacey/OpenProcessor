"""The durable per-source queue: what a source has seen, and what is left to ingest.

Every trigger of a source (a watcher scan, a webhook event, a cron fire, a manual run) does the
same thing here: :meth:`SourceQueue.enqueue` rows into ``ingest_source_files``. A bulk run drains
the ``pending`` rows. The primary key is the idempotency of the whole feature: a replayed event,
a repeated scan and an overlapping pass insert nothing.

A file's key names one version of it: for an S3 source the object reference
(``s3://bucket/key#etag=...``), for a folder source the path and ``"<mtime_ns>:<size>"`` joined
by :data:`VERSION_SEP`, so a file rewritten in place is queued again. :func:`path_of` takes the
part a run sends to the ingest route.

Calls are blocking and open a short transaction each, so a scan thread, a webhook handler and a
run can all use one queue; an async caller runs them with ``asyncio.to_thread``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.db.repos import ingest_sources as repo
from src.db.session import sync_project_tx, sync_session_factory
from src.services.curation.bulk_ingest.object_store import ObjectRef
from src.services.curation.ingest_models import (
    ERROR_KIND_BUSY,
    ERROR_KIND_DATABASE_BUSY,
    ERROR_KIND_ENGINE_UNAVAILABLE,
    ERROR_KIND_INCOMPLETE_WRITE,
    TRANSIENT_ERROR_KINDS,
)


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Iterator
    from contextlib import AbstractContextManager

    from sqlalchemy.orm import Session

MAX_ATTEMPTS = repo.MAX_ATTEMPTS
VERSION_SEP = '\x1f'
# Failures that retrying cannot fix.
TERMINAL_ERROR_KINDS = frozenset(
    {'gone', 'too_large', 'unsupported_type', 'empty', 'decode_failed', 'unservable_path'}
)
STATES = ('pending', 'claimed', 'done', 'failed', 'dead')

# How long a file waits after a transient failure (``TRANSIENT_ERROR_KINDS``), in seconds. The
# kind decides because the causes clear at different speeds:
# - the engine is ready again 2.7 s after an engine restart and about 30 s after a model-server
#   restart plus its reconnect (plan 13a, E1/R3): 30 s;
# - a busy connection pool clears in seconds and every API worker shares it: 10 s;
# - a batch that outlived the sender's retries (about 2.5 minutes of backoff) means a longer
#   outage: 60 s on top of that;
# - an incomplete photo is repaired by the next ingest once its image is older than
#   ``ingest_writer.REPAIR_AFTER_S`` (5,047 s) or its writer marked the failure; each retry is one
#   lookup, so the longest pause (about 17 tries at most) costs little: 300 s.
# The ``Retry-After`` the API sent raises a pause, and no pause passes ``MAX_RETRY_PAUSE_S``.
RETRY_PAUSE_S = {
    ERROR_KIND_ENGINE_UNAVAILABLE: 30.0,
    ERROR_KIND_DATABASE_BUSY: 10.0,
    ERROR_KIND_BUSY: 60.0,
    ERROR_KIND_INCOMPLETE_WRITE: 300.0,
}
MAX_RETRY_PAUSE_S = 300.0


@dataclass(frozen=True)
class QueueEntry:
    key: str
    etag: str
    version: str | None = None
    size: int | None = None


def path_of(file_key: str) -> str:
    """What a run sends to the ingest route for ``file_key``: the path or object reference."""
    return file_key.partition(VERSION_SEP)[0]


def version_of(file_key: str) -> str:
    """The etag part of ``file_key``: the object's ETag, or the folder file's ``mtime_ns:size``."""
    if file_key.startswith('s3://'):
        try:
            return ObjectRef.parse(file_key).etag
        except ValueError:
            return ''
    return file_key.partition(VERSION_SEP)[2]


def file_key_for(entry: QueueEntry, *, backend: str, bucket: str = '') -> str:
    """The ``ingest_source_files`` key of one version of a file (see the module docstring)."""
    if backend == 's3':
        return ObjectRef(bucket, entry.key, entry.etag, entry.version, entry.size).to_uri()
    return f'{entry.key}{VERSION_SEP}{entry.etag}' if entry.etag else entry.key


class SourceQueue:
    """One source's files and scan state in PostgreSQL."""

    def __init__(
        self, project_id: uuid.UUID, source_id: uuid.UUID, *, backend: str, bucket: str = ''
    ):
        self.project_id = project_id
        self.source_id = source_id
        self._s3 = backend == 's3'
        self._bucket = bucket

    def _tx(self) -> AbstractContextManager[Session]:
        return _scoped(self.project_id)

    def file_key(self, entry: QueueEntry) -> str:
        return file_key_for(entry, backend='s3' if self._s3 else 'folder', bucket=self._bucket)

    # ------------------------------------------------------------------ writes

    def enqueue(self, entries: Iterable[QueueEntry]) -> tuple[int, int]:
        """Insert the entries not yet known; returns ``(queued, duplicates)``. One transaction,
        so a webhook batch is all-or-nothing."""
        keys = [self.file_key(e) for e in entries]
        if not keys:
            return 0, 0
        with self._tx() as session:
            return repo.enqueue(session, self.source_id, keys)

    def enqueue_keys(self, file_keys: list[str]) -> tuple[int, int]:
        """Insert keys that are already file keys (a manifest's paths)."""
        if not file_keys:
            return 0, 0
        with self._tx() as session:
            return repo.enqueue(session, self.source_id, file_keys)

    def mark_done(self, job_id: uuid.UUID, rows: Iterable[tuple[str, str | None]]) -> None:
        """Settle ``(file_key, image_id)`` rows as done."""
        with self._tx() as session:
            repo.mark_done(session, self.source_id, job_id, list(rows))

    def retry_dead(self) -> int:
        with self._tx() as session:
            return repo.retry_dead(session, self.source_id)

    def release_claims(self, job_id: uuid.UUID | None = None) -> int:
        with self._tx() as session:
            return repo.release_claims(session, self.source_id, job_id)

    def requeue_all(self) -> int:
        with self._tx() as session:
            return repo.requeue_all(session, self.source_id)

    # ------------------------------------------------------------------- reads

    def counts(self) -> dict[str, int]:
        with self._tx() as session:
            return repo.counts(session, self.source_id)

    def errors(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """The most recent failed rows, for the status view."""
        with self._tx() as session:
            return repo.errors(session, self.source_id, limit)

    # -------------------------------------------------------------------- meta

    def meta_get(self, name: str, default: Any = None) -> Any:
        with self._tx() as session:
            return repo.state_get(session, self.source_id, name, default)

    def meta_all(self) -> dict[str, Any]:
        """The whole scan and trigger state."""
        with self._tx() as session:
            row = repo.get(session, self.source_id)
        return {} if row is None else row.scan_state

    def meta_set(self, name: str, value: Any) -> None:
        with self._tx() as session:
            repo.state_set(session, self.source_id, name, value)

    def meta_update(self, values: dict[str, Any]) -> None:
        with self._tx() as session:
            repo.state_update(session, self.source_id, values)

    def add_counters(self, amounts: dict[str, int]) -> None:
        """Add to several counters in one transaction (webhook totals)."""
        with self._tx() as session:
            repo.state_add(session, self.source_id, amounts)


def retry_pause_s(error_kind: str, retry_after_s: float | None = None) -> float | None:
    """Seconds a file waits after a failure of this kind, or ``None`` when the kind is not
    transient (the failure then counts as an attempt, as before)."""
    if error_kind not in TRANSIENT_ERROR_KINDS:
        return None
    return min(MAX_RETRY_PAUSE_S, max(RETRY_PAUSE_S[error_kind], retry_after_s or 0.0))


def failure_row(
    file_key: str, error: str, error_kind: str, retry_after_s: float | None = None
) -> repo.Failure:
    """The ledger's row for a failed file: terminal kinds are dead at once, transient kinds wait
    and keep their attempts, any other kind counts one attempt."""
    return repo.Failure(
        file_key,
        repo.error_text(error, error_kind),
        terminal=error_kind in TERMINAL_ERROR_KINDS,
        retry_in_s=retry_pause_s(error_kind, retry_after_s),
    )


@contextmanager
def _scoped(project_id: uuid.UUID) -> Iterator[Session]:
    """A session in the project's scope; commits on a clean exit."""
    with sync_session_factory()() as session, sync_project_tx(session, project_id):
        yield session


__all__ = [
    'MAX_ATTEMPTS',
    'MAX_RETRY_PAUSE_S',
    'RETRY_PAUSE_S',
    'STATES',
    'TERMINAL_ERROR_KINDS',
    'VERSION_SEP',
    'QueueEntry',
    'SourceQueue',
    'failure_row',
    'path_of',
    'retry_pause_s',
    'version_of',
]

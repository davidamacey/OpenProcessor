"""The on-disk record of dataset imports (W10.11).

One directory per import under the bound project's imports dir
(``OP_DATASET_IMPORTS_DIR/projects/<slug>/<import_id>/``, on the shared
jobs volume so every ``yolo-api`` worker process sees it):

``request.json`` (the request as accepted), ``mapping.json`` (the resolved
mapping with real class ids, plus the pinned region profile: a resume in
another process consumes THIS, never a re-resolution), ``scan.json`` (the
job's scan summary), ``state.json`` / ``heartbeat`` / ``cancel.flag`` (the
:class:`~src.services.curation.file_job.FileJob` convention),
``ledger/<chunk:06d>.jsonl`` (one event line per image, write-ahead),
``chunks_done.jsonl`` and ``issues.jsonl`` (job-time issues, unsampled).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.project_context import mark_project_dir, project_jobs_dir
from src.services.curation.dataset_import.limits import imports_base_dir
from src.services.curation.file_job import FileJob


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


IMPORT_ID_RE = re.compile(r'^imp_\d{8}T\d{6}_[0-9a-f]{8}$')

ACTIVE_STATUSES = frozenset({'queued', 'running', 'paused_backpressure', 'undoing'})
COMPLETED_STATUSES = frozenset({'completed', 'completed_with_errors'})
RESUMABLE_STATUSES = frozenset({'interrupted', 'failed', 'cancelled'})
UNDOABLE_STATUSES = frozenset(
    {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted', 'undone'}
)


def imports_root() -> Path:
    """The bound project's imports dir (raises ``ProjectNotBound`` unbound)."""
    return mark_project_dir(project_jobs_dir(imports_base_dir()))


def valid_import_id(value: str) -> bool:
    """Only an id this module could have generated names a directory: a
    path-shaped id (``..``, ``/``) never reaches the filesystem."""
    return bool(IMPORT_ID_RE.fullmatch(value))


def new_import_id(import_key: str, *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime('%Y%m%dT%H%M%S')
    return f'imp_{stamp}_{import_key[:8]}'


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, default=str), encoding='utf-8')
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a torn last line from a crash: the write-ahead row is redone
        if isinstance(row, dict):
            yield row


class ImportStore:
    """One import's directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.job = FileJob(directory)

    @property
    def import_id(self) -> str:
        return self.directory.name

    def repaired_state(self) -> dict[str, Any]:
        """The job state, with a run left active by a dead worker rewritten
        to ``interrupted`` first (so it can be resumed or undone)."""
        return self.job.repair_if_stale(ACTIVE_STATUSES, error_prefix='dataset import')

    # ------------------------------------------------------------- documents

    def write_request(self, payload: dict[str, Any]) -> None:
        _atomic_json(self.directory / 'request.json', payload)

    def read_request(self) -> dict[str, Any]:
        return _read_json(self.directory / 'request.json') or {}

    def write_mapping(self, payload: dict[str, Any]) -> None:
        _atomic_json(self.directory / 'mapping.json', payload)

    def read_mapping(self) -> dict[str, Any]:
        return _read_json(self.directory / 'mapping.json') or {}

    def write_scan(self, payload: dict[str, Any]) -> None:
        _atomic_json(self.directory / 'scan.json', payload)

    def read_scan(self) -> dict[str, Any]:
        return _read_json(self.directory / 'scan.json') or {}

    # ----------------------------------------------------------------- ledger

    def _ledger_file(self, chunk: int) -> Path:
        return self.directory / 'ledger' / f'{chunk:06d}.jsonl'

    def append_ledger(self, chunk: int, row: dict[str, Any]) -> None:
        """Append one event line and fsync it: the write-ahead row must be
        durable before the docs it describes are touched."""
        path = self._ledger_file(chunk)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, default=str) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    def chunk_rows(self, chunk: int) -> dict[str, dict[str, Any]]:
        """The latest event per image in ``chunk``, keyed by ``rel_path``."""
        latest: dict[str, dict[str, Any]] = {}
        for row in _iter_jsonl(self._ledger_file(chunk)):
            rel = row.get('rel_path')
            if isinstance(rel, str):
                latest[rel] = row
        return latest

    def ledger_rows(self) -> Iterator[dict[str, Any]]:
        """Every image's latest event, in chunk then scan order."""
        ledger_dir = self.directory / 'ledger'
        if not ledger_dir.is_dir():
            return
        for path in sorted(ledger_dir.glob('*.jsonl')):
            latest: dict[str, dict[str, Any]] = {}
            for row in _iter_jsonl(path):
                rel = row.get('rel_path')
                if isinstance(rel, str):
                    latest[rel] = row
            yield from latest.values()

    def drop_ledger(self) -> None:
        """Remove the ledger (an undo that finished consumes it)."""
        ledger_dir = self.directory / 'ledger'
        if ledger_dir.is_dir():
            for path in ledger_dir.glob('*.jsonl'):
                with contextlib.suppress(OSError):
                    path.unlink()

    # ------------------------------------------------------------ chunk marks

    def mark_chunk_done(self, chunk: int, report: dict[str, Any]) -> None:
        path = self.directory / 'chunks_done.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({'chunk': chunk, 'report': report}, default=str) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    def chunks_done(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for row in _iter_jsonl(self.directory / 'chunks_done.jsonl'):
            if isinstance(row.get('chunk'), int):
                out[row['chunk']] = row.get('report') or {}
        return out

    # ----------------------------------------------------------------- issues

    def append_issues(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.directory / 'issues.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + '\n')

    def issues(self) -> list[dict[str, Any]]:
        return list(_iter_jsonl(self.directory / 'issues.jsonl'))


def open_store(import_id: str) -> ImportStore | None:
    """The store for ``import_id``, or ``None`` when the id is malformed or
    names no import in the bound project."""
    if not valid_import_id(import_id):
        return None
    directory = imports_root() / import_id
    if not directory.is_dir():
        return None
    store = ImportStore(directory)
    store.repaired_state()
    return store


def list_stores() -> list[ImportStore]:
    """Every import of the bound project, newest first (ids carry a UTC
    timestamp, so name order is time order)."""
    root = imports_root()
    if not root.is_dir():
        return []
    stores = [ImportStore(p) for p in root.iterdir() if p.is_dir() and valid_import_id(p.name)]
    for store in stores:
        store.repaired_state()
    return sorted(stores, key=lambda s: s.import_id, reverse=True)


def latest_with_key(import_key: str) -> ImportStore | None:
    """The newest import whose ``import_key`` is ``import_key``."""
    for store in list_stores():
        if store.job.read().get('import_key') == import_key:
            return store
    return None


def active_import() -> ImportStore | None:
    """The one live import of the bound project, if any (a stale heartbeat
    is not live)."""
    for store in list_stores():
        if store.job.is_live(ACTIVE_STATUSES):
            return store
    return None


def running_import_ids() -> list[tuple[str, str | None]]:
    """``[(import_id, started_at)]`` of live imports in the bound project."""
    return [
        (store.import_id, store.job.read().get('started_at'))
        for store in list_stores()
        if store.job.is_live(ACTIVE_STATUSES)
    ]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    'ACTIVE_STATUSES',
    'COMPLETED_STATUSES',
    'IMPORT_ID_RE',
    'RESUMABLE_STATUSES',
    'UNDOABLE_STATUSES',
    'ImportStore',
    'active_import',
    'imports_root',
    'latest_with_key',
    'list_stores',
    'new_import_id',
    'now_iso',
    'open_store',
    'running_import_ids',
    'valid_import_id',
]

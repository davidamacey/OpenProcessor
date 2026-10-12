"""The on-disk record of bulk-ingest runs.

One directory per run under the bound project's bulk-ingest dir
(``OP_BULK_INGEST_DIR/projects/<slug>/<run_id>/``, on the shared jobs volume):

``request.json`` (the source and options as accepted), ``state.json`` /
``heartbeat`` / ``cancel.flag`` (the :class:`~src.services.curation.file_job.FileJob`
convention), ``done.jsonl`` (one line per image that reached a terminal good
state, written after the pipeline stored it: what a resume skips) and
``failures.jsonl`` (this attempt's failed images; a resume retries them, so the
file starts empty on each attempt).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.project_context import mark_project_dir, project_jobs_dir
from src.services.curation.bulk_ingest.limits import bulk_ingest_base_dir
from src.services.curation.file_job import FileJob


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

RUN_ID_RE = re.compile(r'^bulk_\d{8}T\d{6}_[0-9a-f]{8}$')

ACTIVE_STATUSES = frozenset({'queued', 'running'})
# A run that finished with failed images is resumed (the ledger skips the rest), not restarted.
RESUMABLE_STATUSES = frozenset({'interrupted', 'failed', 'cancelled', 'completed_with_errors'})

OUTCOME_INGESTED = 'ingested'
OUTCOME_DUPLICATE = 'duplicate'


def runs_root() -> Path:
    """The bound project's run dir (raises ``ProjectNotBound`` when unbound)."""
    return mark_project_dir(project_jobs_dir(bulk_ingest_base_dir()))


def valid_run_id(value: str) -> bool:
    """Only an id this module could have generated names a directory."""
    return bool(RUN_ID_RE.fullmatch(value))


def new_run_id(*, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime('%Y%m%dT%H%M%S')
    return f'bulk_{stamp}_{uuid.uuid4().hex[:8]}'


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
            continue  # a torn last line from a crash: that image is redone
        if isinstance(row, dict):
            yield row


class BulkStore:
    """One run's directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.job = FileJob(directory)

    @property
    def run_id(self) -> str:
        return self.directory.name

    def repaired_state(self) -> dict[str, Any]:
        """The job state, with a run left active by a dead worker rewritten to
        ``interrupted`` first (so it can be resumed)."""
        return self.job.repair_if_stale(ACTIVE_STATUSES, error_prefix='bulk ingest')

    def write_request(self, payload: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / 'request.json.tmp'
        tmp.write_text(json.dumps(payload), encoding='utf-8')
        tmp.replace(self.directory / 'request.json')

    def read_request(self) -> dict[str, Any]:
        try:
            raw = json.loads((self.directory / 'request.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    # ------------------------------------------------------------------ ledger

    def append_done(self, rows: list[tuple[str, str]]) -> None:
        """Append ``(path, outcome)`` lines and fsync: a path is in the ledger only
        after the pipeline stored it."""
        if not rows:
            return
        with (self.directory / 'done.jsonl').open('a', encoding='utf-8') as fh:
            for path, outcome in rows:
                fh.write(json.dumps({'p': path, 'o': outcome}) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    def done(self) -> tuple[set[str], int, int]:
        """``(paths, ingested, duplicates)`` over the whole ledger."""
        paths: set[str] = set()
        ingested = duplicates = 0
        for row in _iter_jsonl(self.directory / 'done.jsonl'):
            path = row.get('p')
            if not isinstance(path, str) or path in paths:
                continue
            paths.add(path)
            if row.get('o') == OUTCOME_DUPLICATE:
                duplicates += 1
            else:
                ingested += 1
        return paths, ingested, duplicates

    def reset_failures(self) -> None:
        (self.directory / 'failures.jsonl').write_text('', encoding='utf-8')

    def append_failures(self, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        with (self.directory / 'failures.jsonl').open('a', encoding='utf-8') as fh:
            for row in rows:
                fh.write(json.dumps(row) + '\n')

    def failures(self, *, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        """``(page, total)`` of this attempt's failures."""
        rows = list(_iter_jsonl(self.directory / 'failures.jsonl'))
        return rows[offset : offset + limit], len(rows)


def open_store(run_id: str) -> BulkStore | None:
    """The store for ``run_id``, or ``None`` when the id is malformed or names no
    run in the bound project."""
    if not valid_run_id(run_id):
        return None
    directory = runs_root() / run_id
    if not directory.is_dir():
        return None
    store = BulkStore(directory)
    store.repaired_state()
    return store


def list_stores() -> list[BulkStore]:
    """Every run of the bound project, newest first (ids carry a UTC timestamp)."""
    root = runs_root()
    if not root.is_dir():
        return []
    stores = [BulkStore(p) for p in root.iterdir() if p.is_dir() and valid_run_id(p.name)]
    for store in stores:
        store.repaired_state()
    return sorted(stores, key=lambda s: s.run_id, reverse=True)


def active_run() -> BulkStore | None:
    """The one live run of the bound project, if any (a stale heartbeat is not live)."""
    for store in list_stores():
        if store.job.is_live(ACTIVE_STATUSES):
            return store
    return None


def running_run_ids() -> list[tuple[str, str | None]]:
    """``[(run_id, started_at)]`` of live runs in the bound project."""
    return [
        (store.run_id, store.job.read().get('started_at'))
        for store in list_stores()
        if store.job.is_live(ACTIVE_STATUSES)
    ]

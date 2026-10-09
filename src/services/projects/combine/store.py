"""The on-disk record of combine jobs.

One directory per job under ``OP_COMBINE_JOBS_DIR`` (shared jobs volume, so
every ``api`` worker process sees it): the W10 import store's layout
(``request.json``, ``mapping.json``, ``state.json`` / ``heartbeat`` /
``cancel.flag``, the write-ahead ledger, ``chunks_done.jsonl``), reused as is.
A combine is global (it acts on several projects), so its directory is not
nested under a project, and its job id names the way back to the target.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.services.curation.dataset_import.store import ImportStore


if TYPE_CHECKING:
    from collections.abc import Iterator

JOB_ID_RE = re.compile(r'^cmb_\d{8}T\d{6}_[0-9a-f]{8}$')

ACTIVE_STATUSES = frozenset({'queued', 'running'})
COMPLETED_STATUSES = frozenset({'completed', 'completed_with_errors'})
RESUMABLE_STATUSES = frozenset({'interrupted', 'cancelled'})


def combine_base_dir() -> Path:
    return Path(os.environ.get('OP_COMBINE_JOBS_DIR', '/jobs/combine'))


def valid_job_id(value: str) -> bool:
    """Only an id this module generated names a directory."""
    return bool(JOB_ID_RE.fullmatch(value))


def new_job_id(*, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime('%Y%m%dT%H%M%S')
    return f'cmb_{stamp}_{uuid.uuid4().hex[:8]}'


def open_job(job_id: str) -> ImportStore | None:
    """The store for ``job_id``, or ``None`` when it is malformed or unknown."""
    if not valid_job_id(job_id):
        return None
    store = ImportStore(combine_base_dir() / job_id)
    return store if store.directory.is_dir() else None


def create_job(job_id: str) -> ImportStore:
    store = ImportStore(combine_base_dir() / job_id)
    store.directory.mkdir(parents=True, exist_ok=True)
    return store


def iter_jobs() -> Iterator[ImportStore]:
    base = combine_base_dir()
    if not base.is_dir():
        return
    for child in sorted(base.iterdir()):
        if child.is_dir() and valid_job_id(child.name):
            yield ImportStore(child)


def involves(state: dict[str, object], slug: str) -> bool:
    return state.get('target') == slug or slug in (state.get('sources') or [])  # type: ignore[operator]


def running_jobs_for(slug: str) -> list[tuple[str, str | None]]:
    """``(job_id, started_at)`` of every live combine job that reads or writes
    ``slug`` (live: an active status with a fresh heartbeat)."""
    out: list[tuple[str, str | None]] = []
    for store in iter_jobs():
        state = store.job.read()
        if involves(state, slug) and store.job.is_live(ACTIVE_STATUSES):
            out.append((store.import_id, state.get('started_at')))  # type: ignore[arg-type]
    return out


def reconcile_orphaned_jobs() -> int:
    """Startup repair: a job left active by a dead process becomes
    ``interrupted`` (resumable). Returns how many were repaired."""
    repaired = 0
    for store in iter_jobs():
        if store.job.reconcile(
            active_statuses=ACTIVE_STATUSES, error_prefix='combine interrupted by a restart'
        ):
            repaired += 1
    return repaired


__all__ = [
    'ACTIVE_STATUSES',
    'COMPLETED_STATUSES',
    'RESUMABLE_STATUSES',
    'combine_base_dir',
    'create_job',
    'involves',
    'iter_jobs',
    'new_job_id',
    'open_job',
    'reconcile_orphaned_jobs',
    'running_jobs_for',
    'valid_job_id',
]

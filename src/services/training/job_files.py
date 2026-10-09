"""Training-job file layout: jobs dir, ids, sentinel paths, atomic JSON I/O.

Split out of :mod:`src.services.training.jobs`. Every on-disk name the trainer
container also reads (``<job_id>.job.json``, ``.status.json``, ``.cancel``,
``.run.log``, ``.manifest.json``) is built here.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import get_curation_config


# =============================================================================
# Constants
# =============================================================================


def _resolve_jobs_dir() -> Path:
    """The bound project's ``train_jobs_dir`` (``<OP_TRAIN_JOBS_DIR>/projects/<slug>``,
    ``default`` included; see ``src.config.projects.resources_for_new``), so
    job.json writes and run listing never cross a project boundary."""
    return get_curation_config().train_jobs_dir


def trainer_root_dir() -> Path:
    """The trainer's own watch root (``OP_TRAIN_JOBS_DIR``, default ``/jobs``).

    One trainer serves every project: it globs each project's dir under
    this root, but writes its process-wide files (``.trainer_capabilities.json``)
    here, not under any project's dir."""
    return Path(os.environ.get('OP_TRAIN_JOBS_DIR', '/jobs'))


# Public for callers that want the default without the env override.
TRAIN_JOBS_DIR = Path('/jobs')

# Heartbeat older than this -> flip status to ``lost`` in the API view.
STALE_HEARTBEAT_SECONDS = 60

# State the trainer writes; ``lost`` is API-side only.
TRAIN_STATES = (
    'queued',
    'starting',
    'running',
    'exporting',
    'finished',
    'failed',
    'cancelled',
    'skipped',
    'lost',
)

# Job-id format: ISO timestamp prefix + tag. Validated with this regex when
# we accept a job-id in path params so we don't end up reading arbitrary
# files via ``..`` injection.
JOB_ID_RE = re.compile(r'^[A-Za-z0-9_.\-:+T]{1,128}$')


# =============================================================================
# File helpers
# =============================================================================


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _slug() -> str:
    """Filesystem-safe ISO timestamp (``2026-05-09T08-30-14``)."""
    return datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%S')


def _job_path(job_id: str) -> Path:
    return _resolve_jobs_dir() / f'{job_id}.job.json'


def _status_path(job_id: str) -> Path:
    return _resolve_jobs_dir() / f'{job_id}.status.json'


def _cancel_path(job_id: str) -> Path:
    return _resolve_jobs_dir() / f'{job_id}.cancel'


def _log_path(job_id: str) -> Path:
    return _resolve_jobs_dir() / f'{job_id}.run.log'


def _manifest_path(job_id: str) -> Path:
    return _resolve_jobs_dir() / f'{job_id}.manifest.json'


def _registry_snapshot_path(job_id: str) -> Path:
    return _resolve_jobs_dir() / f'{job_id}.registry_snapshot.json'


def _validate_job_id(job_id: str) -> None:
    """Reject obviously-bad ids before they hit the filesystem."""
    if not JOB_ID_RE.match(job_id):
        msg = f'invalid job_id: {job_id!r}'
        raise ValueError(msg)


async def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically (tmp + rename). Runs in a thread to keep
    the event loop unblocked."""

    def _do() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)

    await asyncio.to_thread(_do)


async def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON or return ``None`` if missing."""

    def _do() -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        return data

    return await asyncio.to_thread(_do)

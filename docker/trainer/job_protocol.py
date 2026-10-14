"""The API <-> trainer job-file protocol, trainer side.

The API never RPCs the trainer: the entire contract is files in a shared
volume. The API-side half lives in :mod:`src.services.training.jobs`; this
module is the other half and must stay wire-compatible with it.

    API     writes  <jobs_dir>/<job_id>.job.json       to start a run
    trainer writes  <jobs_dir>/<job_id>.status.json    while it runs
    API     writes  <jobs_dir>/<job_id>.cancel         to cancel
    trainer appends <jobs_dir>/<job_id>.run.log        for the log-tail route
    trainer writes  <jobs_dir>/<job_id>.manifest.json  at the end

Everything here is about those files -- parsing a submitted job, publishing
status + heartbeats, tee-ing the run log, writing the lineage manifest, and
deciding which jobs are still pending. The training loop itself lives in
:mod:`trainer`, dataset preparation in :mod:`dataset_prep`, and cross-job
campaign policy in :mod:`campaign`.
"""

from __future__ import annotations

import contextlib
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from job_spec import _job_id_from_path, _read_json, _status_path_for
from job_status import TERMINAL_STATES, _atomic_write_json, _utcnow_iso
from logutil import get_logger


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


logger = get_logger('trainer.job_protocol')


# ---------------------------------------------------------------------------
# Run-log tee
# ---------------------------------------------------------------------------


@contextmanager
def tee_to_run_log(run_log_path: Path) -> Iterator[None]:
    """Tee stdout + stderr into ``run_log_path``.

    This is what makes ``GET {api_prefix}/train/log/tail/{job_id}`` show
    Ultralytics' own progress output.
    """
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = run_log_path.open('a', encoding='utf-8', buffering=1)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    class _Tee:
        def __init__(self, *streams: Any) -> None:
            self._streams = streams

        def write(self, data: str) -> int:
            for s in self._streams:
                with contextlib.suppress(Exception):
                    s.write(data)
                    s.flush()
            return len(data)

        def flush(self) -> None:
            for s in self._streams:
                with contextlib.suppress(Exception):
                    s.flush()

        def isatty(self) -> bool:
            return False

    try:
        sys.stdout = _Tee(orig_stdout, fp)
        sys.stderr = _Tee(orig_stderr, fp)
        yield
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        with contextlib.suppress(Exception):
            fp.close()


# ---------------------------------------------------------------------------
# Work queue: which job files still need running
# ---------------------------------------------------------------------------


def _is_terminal(status_path: Path) -> bool:
    """True when a job's status.json already shows a terminal state."""
    if not status_path.is_file():
        return False
    try:
        return _read_json(status_path).get('state') in TERMINAL_STATES
    except (OSError, ValueError):
        return False


def list_pending_jobs(jobs_dir: Path) -> list[Path]:
    """Find ``*.job.json`` files that haven't reached a terminal state,
    across every project's own dir under ``<jobs_dir>/projects/<slug>``
    (docs/design/openprocessor_internal/projects_plan.md §5.3;
    P1R §6.1/D-A: ``default`` nests the same as every other project --
    see ``src.config.projects.resources_for_new``). The top-level
    ``jobs_dir`` glob is kept for a job file dropped directly there by
    hand; nothing in this codebase writes one.

    Sorted by mtime across every project's dir, so a project's job never
    starves another's (still one run at a time, FIFO) -- matching the
    order the API writes a campaign's runs in.
    """
    if not jobs_dir.is_dir():
        return []
    candidates = list(jobs_dir.glob('*.job.json')) + list(jobs_dir.glob('projects/*/*.job.json'))
    return [
        p
        for p in sorted(candidates, key=lambda x: x.stat().st_mtime)
        if not _is_terminal(_status_path_for(p))
    ]


def reject_job(job_path: Path, exc: Exception) -> None:
    """Write a terminal ``failed`` status for an unparseable job file.

    Without this the watcher would re-read the same broken file forever.
    """
    status_path = _status_path_for(job_path)
    code = getattr(exc, 'code', 'validation_failed')
    payload = {
        'job_id': _job_id_from_path(job_path),
        'state': 'failed',
        'error': f'validation: {exc}',
        'error_code': code,
        'finished_at': _utcnow_iso(),
        'heartbeat_at': _utcnow_iso(),
    }
    try:
        _atomic_write_json(status_path, payload)
    except OSError:
        logger.exception('failed to write rejection status', path=str(status_path))

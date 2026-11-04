"""The file-backed job-runner convention, as one small class.

``state.json`` (atomic temp+rename), ``heartbeat`` and ``cancel.flag`` under one
directory, on the shared ``/jobs`` volume: every ``api`` worker process sees the same
files, so status and cancel are correct whichever process a request lands on.

Singleton start is not decided here but by the job lease (``src/clients/job_lease.py``).
Only the lease holder writes ``state.json``: a read-modify-write needs no file lock, and a
reader never writes (:meth:`served_state` derives ``interrupted`` for a dead run).

Used by the dataset-import, reprocess, probe, item-score, selection,
embedding-viz and auto-label jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.domain.job_lease import LEASE_TTL_S
from src.services.curation.job_reconcile import reconcile_stale_running
from src.utils.atomic_write import atomic_write_json


if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

HEARTBEAT_STALE_S = LEASE_TTL_S


@dataclass(frozen=True)
class FileJob:
    """One job's on-disk state directory."""

    directory: Path

    @property
    def state_file(self) -> Path:
        return self.directory / 'state.json'

    @property
    def heartbeat_file(self) -> Path:
        return self.directory / 'heartbeat'

    @property
    def cancel_file(self) -> Path:
        return self.directory / 'cancel.flag'

    def read(self) -> dict[str, Any]:
        """The current state; ``{}`` when the file is missing or unreadable
        (a half-written file is impossible: writes are atomic renames). A busy
        check must not trust that ``{}``: see :meth:`state_problem`."""
        try:
            raw = json.loads(self.state_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def state_problem(self) -> str | None:
        """Why the state file cannot be read, or ``None`` when it is missing (a job
        that never ran) or readable. An unreadable state says nothing about whether
        a job is running."""
        try:
            raw = json.loads(self.state_file.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            return f'{type(exc).__name__}: {exc}'
        return None if isinstance(raw, dict) else 'the state is not a JSON object'

    def write(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_file, state)

    def update(self, **fields: Any) -> dict[str, Any]:
        """Read-modify-write of the state; only the lease holder calls it."""
        state = self.read()
        state.update(fields)
        self.write(state)
        return state

    def touch_heartbeat(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.heartbeat_file.touch()

    def heartbeat_age(self) -> float | None:
        try:
            return max(0.0, time.time() - self.heartbeat_file.stat().st_mtime)
        except OSError:
            return None

    def is_live(self, active_statuses: frozenset[str]) -> bool:
        """An active status with a heartbeat no older than
        :data:`HEARTBEAT_STALE_S`. No heartbeat yet means the task has not
        ticked once: still live. A state file that exists but cannot be read
        counts as live: the busy check fails closed rather than start a second
        job beside one it cannot see."""
        problem = self.state_problem()
        if problem is not None:
            logger.warning('job_state_unreadable', path=str(self.state_file), problem=problem)
            return True
        state = self.read()
        return state.get('status') in active_statuses and not self._is_stale(state, active_statuses)

    def served_state(self, active_statuses: frozenset[str], *, error_prefix: str) -> dict[str, Any]:
        """The current state as a reader should see it: a run left active by a dead process
        (an active status whose heartbeat went stale) is reported ``interrupted``, so it can
        be resumed or undone. The view only: a reader never writes state. Persisting the
        repair belongs to whoever takes the lease over (a resume, the startup reconcile). A
        job that has not ticked once yet is not stale."""
        state = self.read()
        if not self._is_stale(state, active_statuses):
            return state
        return {
            **state,
            'status': 'interrupted',
            'error': state.get('error') or f'{error_prefix} heartbeat stale',
            'finished_at': state.get('finished_at') or datetime.now(UTC).isoformat(),
        }

    def _is_stale(self, state: dict[str, Any], active_statuses: frozenset[str]) -> bool:
        if state.get('status') not in active_statuses:
            return False
        age = self.heartbeat_age()
        return age is not None and age > HEARTBEAT_STALE_S

    def request_cancel(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cancel_file.touch()

    def cancel_requested(self) -> bool:
        return self.cancel_file.exists()

    def clear_signals(self) -> None:
        """Drop the cancel flag and heartbeat before a (re)start."""
        for path in (self.cancel_file, self.heartbeat_file):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    def reconcile(self, *, active_statuses: frozenset[str], error_prefix: str) -> bool:
        """Startup repair: a state left active by a dead process becomes
        ``interrupted`` (see :mod:`~src.services.curation.job_reconcile`)."""
        return reconcile_stale_running(
            self.state_file,
            self.heartbeat_file,
            stale_s=HEARTBEAT_STALE_S,
            error_prefix=error_prefix,
            active_statuses=active_statuses,
        )


async def heartbeat_ticker(job: FileJob) -> None:
    """Keep the heartbeat fresh for the whole run, not only between steps."""
    while True:
        await asyncio.sleep(10.0)
        job.touch_heartbeat()


__all__ = ['HEARTBEAT_STALE_S', 'FileJob', 'heartbeat_ticker']

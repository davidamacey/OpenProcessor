"""The file-backed job-runner convention, as one small class.

``state.json`` (atomic temp+rename), ``heartbeat``, ``cancel.flag`` and a
cross-process start lock under one directory, on the shared ``/jobs``
volume: every ``api`` worker process sees the same files, so status,
cancel and singleton start are correct whichever process a request lands
on (the multi-worker fix ``probe_job`` documents).

Used by the dataset-import, reprocess, probe, item-score, selection,
embedding-viz and auto-label jobs.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.services.curation.job_reconcile import reconcile_stale_running


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

HEARTBEAT_STALE_S = 30.0
HEARTBEAT_TICK_S = 10.0


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
        (a half-written file is impossible: writes are atomic renames)."""
        try:
            raw = json.loads(self.state_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def write(self, state: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, default=str), encoding='utf-8')
        tmp.replace(self.state_file)

    @contextlib.contextmanager
    def _state_lock(self) -> Iterator[None]:
        """Blocking cross-process lock around a read-modify-write of the state
        (held for file operations only, never across a job's work)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.directory / 'state.lock', os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def update(self, **fields: Any) -> dict[str, Any]:
        with self._state_lock():
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
        ticked once: still live."""
        state = self.read()
        return state.get('status') in active_statuses and not self._is_stale(state, active_statuses)

    def repair_if_stale(
        self, active_statuses: frozenset[str], *, error_prefix: str
    ) -> dict[str, Any]:
        """The current state, after rewriting one left active by a dead
        process (an active status whose heartbeat went stale) to
        ``interrupted``. The lazy twin of :meth:`reconcile`: it runs on
        whatever reads the job, so a worker killed and restarted inside the
        liveness window (which the startup reconcile leaves alone) is still
        resumable once its heartbeat has aged out. A job that has not
        ticked once yet is not stale. The write re-checks staleness under the
        state lock, so a claim that lands in between
        (resume rewriting the state to ``queued``) is never overwritten with
        ``interrupted``."""
        state = self.read()
        if not self._is_stale(state, active_statuses):
            return state  # the common read: no lock, nothing created
        with self._state_lock():
            state = self.read()
            if not self._is_stale(state, active_statuses):
                return state
            state.update(
                status='interrupted',
                error=state.get('error') or f'{error_prefix} heartbeat stale',
                finished_at=datetime.now(UTC).isoformat(),
            )
            self.write(state)
            return state

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
        await asyncio.sleep(HEARTBEAT_TICK_S)
        job.touch_heartbeat()


__all__ = ['HEARTBEAT_STALE_S', 'HEARTBEAT_TICK_S', 'FileJob', 'heartbeat_ticker']

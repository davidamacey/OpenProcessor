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
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.domain.job_lease import LEASE_TTL_S
from src.services.curation.job_reconcile import reconcile_stale_running
from src.utils.atomic_write import atomic_write_json


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = get_logger(__name__)

_FENCES: dict[Path, Callable[[], bool]] = {}


class LeaseLostError(RuntimeError):
    """A job tried to write its state after its lease was taken over or went unrenewed."""


def register_fence(directory: Path, may_write: Callable[[], bool]) -> None:
    """Fence ``directory``'s state writes in this process: :meth:`FileJob.write` raises
    :class:`LeaseLostError` while ``may_write()`` is false. Set by the lease holder when it
    starts a job (:class:`~src.services.curation.lease_heartbeat.LeaseHeartbeat`)."""
    _FENCES[directory] = may_write


def clear_fence(directory: Path) -> None:
    _FENCES.pop(directory, None)


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
        (a half-written file is impossible: writes are atomic renames)."""
        try:
            raw = json.loads(self.state_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def write(self, state: dict[str, Any]) -> None:
        fence = _FENCES.get(self.directory)
        if fence is not None and not fence():
            msg = f'the lease of the job in {self.directory.name} is lost; refusing to write'
            raise LeaseLostError(msg)
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

    def cancel_requested(self) -> bool:
        return self.cancel_file.exists()

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


__all__ = [
    'HEARTBEAT_STALE_S',
    'FileJob',
    'LeaseLostError',
    'clear_fence',
    'heartbeat_ticker',
    'register_fence',
]

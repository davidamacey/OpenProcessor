"""Shared startup-reconciliation helper for the curation subsystem's
file-backed, in-process job runners.

``item_scores/job.py``, ``selection/job.py`` and ``embedding_viz.py`` each
run their background work as an ``asyncio.Task`` tracked only by a
module-level global (``_active_task``) that does not survive a process
restart. Their ``state.json`` is otherwise durable (atomic temp+rename
writes to a shared volume), so killing the process mid-job leaves that
file stuck reporting ``status='running'`` forever unless something
rewrites it.

Each of those modules' own ``get_state()`` already repairs a stale
heartbeat *lazily*, the next time anything polls status, by rewriting
``status='failed'``. This helper performs the same detection
*proactively*, once, at process startup (see ``src.main``'s lifespan),
so an orphaned run is visible immediately rather than only on next poll.
It deliberately writes a distinct terminal status, ``'interrupted'``,
so a caller can tell "this run was cut off by a restart" apart from
"this run's own code raised an exception" (``'failed'``) — the two are
otherwise indistinguishable from the state file alone once both are
possible outcomes of the same on-disk shape.

This module never invents a new write path: it does the exact same
read-modify-atomic-write-via-temp-file dance every job module here
already does, on the same file. There is no import from any of the
three job modules — passing plain ``Path`` objects keeps this a
leaf dependency they can each import without a cycle.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


def reconcile_stale_running(
    state_file: Path,
    heartbeat_file: Path,
    *,
    stale_s: float,
    error_prefix: str,
) -> bool:
    """Repair a ``state.json`` left at ``status='running'`` by a process
    that no longer exists.

    Called once at startup, before this process has scheduled any job of
    its own — at that point nothing legitimate can be 'running' yet, so
    a missing heartbeat file is treated the same as a stale one (unlike
    the modules' own live ``_is_busy()`` checks, which treat "no
    heartbeat yet" as "hasn't ticked once, still busy" because a
    genuinely fresh job might not have written one yet).

    Returns True if the file was rewritten.
    """
    try:
        raw = json.loads(state_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if raw.get('status') != 'running':
        return False

    try:
        mtime = heartbeat_file.stat().st_mtime
        age: float | None = max(0.0, time.time() - mtime)
    except (FileNotFoundError, OSError):
        age = None

    if age is not None and age <= stale_s:
        # A heartbeat this fresh at startup means the on-disk state
        # predates this process (e.g. a shared volume reused across a
        # fast restart) but is still within the liveness window --
        # leave it for the modules' own get_state() staleness check to
        # decide once nothing has actually touched it for stale_s.
        return False

    raw['status'] = 'interrupted'
    raw['error'] = raw.get('error') or (
        f'{error_prefix} heartbeat stale ({age:.1f}s ago)'
        if age is not None
        else f'{error_prefix} heartbeat missing (process restarted mid-run)'
    )
    raw['finished_at'] = raw.get('finished_at') or time.time()

    tmp = state_file.with_suffix('.tmp')
    tmp.write_text(json.dumps(raw, default=str))
    tmp.replace(state_file)
    return True


__all__ = ['reconcile_stale_running']

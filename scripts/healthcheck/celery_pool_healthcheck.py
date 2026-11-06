#!/usr/bin/env python3
"""Celery prefork-pool healthcheck: catch a pool that is not running tasks.

Ported from OpenTranscribe ``backend/scripts/celery_pool_healthcheck.py`` (AGPL-3.0, the same
author, a public project), adapted to this repository's app and queue layout.

``celery inspect ping`` is answered by the worker's MainProcess whether or not its forked child
pool is alive, so a wedged prefork pool still looks healthy to Docker. ``inspect stats`` is
answered by the same MainProcess but reports pool state. Two failures are checked:

1. Dead pool: ``pool.processes`` is empty.
2. Respawn storm: ``pool.processes`` is non-empty but the whole pid set was replaced since the
   previous probe while the worker's accepted-task counter (``stats["total"]``) did not move.
   Legitimate churn under ``--max-tasks-per-child`` replaces pids *because* tasks completed, so
   the counter advances. Detection needs memory, so one sample is kept in the temp directory;
   a missing or stale sample is judged healthy rather than guessed at.

The pool type is read from ``pool.implementation``, so the script is safe to point at a worker
of any pool: a non-prefork pool lives inside the MainProcess that just answered.

Review R-38: the stats request is sent with ``destination=[this node]`` (``-d``), so a probe
never fans out to every replica. The argument is the node name (``jobs@host``) or just the
prefix (``jobs``), which is completed with this container's hostname.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Celery's own reply window is 1 s, which ``inspect stats`` routinely misses on a busy worker.
INSPECT_REPLY_TIMEOUT_S = 5.0
# Must stay below the compose ``healthcheck.timeout`` or Docker kills the probe first.
INSPECT_TIMEOUT_S = 8
# The probe interval is 30 s, so a sample survives four missed probes.
SAMPLE_STALE_AFTER_S = 150.0
DEFAULT_APP = 'src.tasks.app'


def node_name(argument: str) -> str:
    """``jobs`` becomes ``jobs@<this hostname>``; a full node name is kept."""
    return argument if '@' in argument else f'{argument}@{socket.gethostname()}'


def _sample_path(worker_name: str) -> Path:
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', worker_name)
    return Path(tempfile.gettempdir()) / f'celery-pool-healthcheck.{safe}.json'


def _read_previous_sample(path: Path) -> dict | None:
    try:
        sample = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(sample, dict) or 'at' not in sample:
        return None
    if time.time() - float(sample['at']) > SAMPLE_STALE_AFTER_S:
        return None
    return sample


def _write_sample(path: Path, pids: list, accepted: int) -> None:
    """Best effort: a probe that raises is itself a false unhealthy."""
    with contextlib.suppress(OSError):
        path.write_text(json.dumps({'at': time.time(), 'pids': pids, 'accepted': accepted}))


def _accepted_task_count(stats: dict) -> int:
    """Monotonic for the life of the MainProcess and not reset when a child is recycled."""
    total = stats.get('total')
    if isinstance(total, dict):
        return sum(v for v in total.values() if isinstance(v, int))
    if isinstance(total, int):
        return total
    return 0


def _fetch_stats(worker_name: str, app: str) -> tuple[dict | None, str]:
    """The worker's ``inspect stats`` reply, or ``(None, why it could not be read)``."""
    celery = shutil.which('celery')
    if celery is None:
        return None, 'the celery executable is not on PATH'
    try:
        proc = subprocess.run(  # nosec B603  fixed argv, no shell; the node name is the container's own
            [
                celery,
                '-A',
                app,
                'inspect',
                '--timeout',
                str(INSPECT_REPLY_TIMEOUT_S),
                'stats',
                '-d',
                worker_name,
                '--json',
            ],
            capture_output=True,
            text=True,
            timeout=INSPECT_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, 'celery inspect stats timed out'
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, f'celery inspect stats failed (rc={proc.returncode}): {proc.stderr.strip()}'
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return None, f'could not parse inspect stats output: {exc}'
    if not payload:
        return None, 'no worker replied to inspect stats'
    return next(iter(payload.values())), ''


def check_pool_has_live_processes(worker_name: str, app: str = DEFAULT_APP) -> tuple[bool, str]:
    """``(healthy, reason)`` for the named worker's pool."""
    stats, problem = _fetch_stats(worker_name, app)
    if stats is None:
        return False, problem
    pool = stats.get('pool')
    if not isinstance(pool, dict) or 'implementation' not in pool:
        # Fail closed on a reply that cannot be read: reporting healthy on unread data is the
        # blind check this script replaces.
        return False, f'inspect stats reported no interpretable pool section: pool={pool!r}'

    implementation = str(pool['implementation'])
    if 'prefork' not in implementation:
        return True, f'non-prefork pool answered ({implementation})'
    processes = pool.get('processes')
    if not processes:
        return False, f'prefork pool has no live child processes: pool={pool!r}'
    return _classify_pool_progress(worker_name, processes, _accepted_task_count(stats))


def _classify_pool_progress(worker_name: str, processes: list, accepted: int) -> tuple[bool, str]:
    """A working pool or a respawn storm, judged against the previous probe."""
    path = _sample_path(worker_name)
    previous = _read_previous_sample(path)
    _write_sample(path, list(processes), accepted)

    live = f'{len(processes)} live child process(es)'
    if previous is None:
        return True, f'{live} (no comparable prior sample; storm check deferred)'
    carried_over = set(processes) & set(previous.get('pids') or [])
    if carried_over:
        return True, f'{live}, {len(carried_over)} carried over from the previous probe'
    if accepted > int(previous.get('accepted', 0)):
        return True, f'{live}, pool fully recycled while accepting tasks (total={accepted})'
    return False, (
        f'prefork respawn storm: pool fully replaced since the previous probe '
        f'(was {previous.get("pids")!r}, now {list(processes)!r}) with no task accepted '
        f'(total stuck at {accepted}): children are being killed before they signal UP'
    )


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        sys.stderr.write('usage: celery_pool_healthcheck.py <node-or-prefix> [celery-app]\n')
        return 2
    app = argv[2] if len(argv) == 3 else DEFAULT_APP
    healthy, reason = check_pool_has_live_processes(node_name(argv[1]), app)
    (sys.stdout if healthy else sys.stderr).write(reason + '\n')
    return 0 if healthy else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))

"""Heartbeat-based liveness checks for the curation background workers.

Replaces the ``pgrep -f <module>`` compose healthchecks (S-2): a process
can be alive under ``pgrep`` while its event loop is deadlocked (e.g.
blocked on a synchronous call, or every worker task crashed and only the
outer ``asyncio.run`` frame is left spinning). A heartbeat file that the
worker's own main loop touches — including while idle — actually proves
forward progress.

Each worker calls :func:`heartbeat_loop` as a sibling task next to its
other lifespan tasks (metrics reporter, producer, consumers/writer). It
writes ``{ts, pid, tasks}`` to ``{OP_HEARTBEAT_DIR}/<name>.json`` every
``interval_s`` seconds. ``tasks`` is a small liveness map the caller
supplies (e.g. ``{'producer': not producer_task.done()}``) — a stalled
sub-task still lets the loop run and touch the file, so plain mtime
alone can't see it; a task value of ``False`` fails the check even with
a fresh file.

The compose healthcheck runs this module as a script:

    python -m src.services.curation.worker_liveness check <name> --max-age 120

exiting 0 (healthy) or 1 (unhealthy), printing the reason either way.

``OP_HEARTBEAT_DIR`` defaults to a container-local tmp path — no mount
is needed since the healthcheck always runs inside the same container
as the worker it's checking.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


logger = get_logger(__name__)

HEARTBEAT_DIR = Path(os.environ.get('OP_HEARTBEAT_DIR', '/tmp/openprocessor_heartbeat'))  # nosec B108 — container-local, no mount needed

DEFAULT_INTERVAL_S = 15.0
DEFAULT_MAX_AGE_S = 120.0


def _heartbeat_path(name: str) -> Path:
    return HEARTBEAT_DIR / f'{name}.json'


def write_heartbeat(name: str, tasks: Mapping[str, bool]) -> None:
    """Atomically write the heartbeat file for worker ``name``.

    ``tasks`` records per-sub-task liveness (e.g. producer/consumer/writer
    task ``not task.done()``); any ``False`` value fails the healthcheck
    even though the file itself is fresh.
    """
    HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {'ts': time.time(), 'pid': os.getpid(), 'tasks': dict(tasks)}
    path = _heartbeat_path(name)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def check_heartbeat(name: str, max_age_s: float) -> tuple[bool, str]:
    """Return ``(healthy, reason)`` for worker ``name``.

    Unhealthy when the file is missing, malformed, older than
    ``max_age_s``, or any ``tasks`` entry is falsy.
    """
    path = _heartbeat_path(name)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return False, f'heartbeat file missing: {path}'
    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f'heartbeat file malformed: {exc}'

    ts = payload.get('ts')
    if not isinstance(ts, int | float):
        return False, 'heartbeat file missing ts'
    age = time.time() - ts
    if age > max_age_s:
        return False, f'heartbeat stale: age={age:.1f}s > max_age={max_age_s:.1f}s'

    tasks = payload.get('tasks') or {}
    dead = [task_name for task_name, alive in tasks.items() if not alive]
    if dead:
        return False, f'task(s) not running: {", ".join(sorted(dead))}'

    return True, f'ok: age={age:.1f}s tasks={list(tasks)}'


async def heartbeat_loop(
    name: str,
    tasks_fn: Callable[[], Mapping[str, bool]],
    stop: asyncio.Event,
    interval_s: float = DEFAULT_INTERVAL_S,
) -> None:
    """Long-lived task: write the heartbeat every ``interval_s`` until ``stop``.

    Designed as a sibling task alongside the worker's other lifespan
    tasks (metrics reporter, producer/consumer/writer): a blocking call
    inside one of those doesn't stop this loop from ticking, so the
    heartbeat only goes stale when the whole event loop actually stalls.
    """
    while not stop.is_set():
        try:
            write_heartbeat(name, tasks_fn())
        except Exception as exc:  # pragma: no cover — defensive, never crash the worker
            logger.warning('heartbeat_write_failed', worker=name, error=str(exc))
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            continue


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='worker_liveness')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='check a worker heartbeat and exit 0/1')
    check.add_argument('name')
    check.add_argument('--max-age', type=float, default=DEFAULT_MAX_AGE_S)
    args = parser.parse_args(argv)

    if args.command == 'check':
        healthy, reason = check_heartbeat(args.name, args.max_age)
        print(reason)
        return 0 if healthy else 1
    return 1  # pragma: no cover — unreachable, argparse enforces valid subcommands


if __name__ == '__main__':
    sys.exit(_cli())

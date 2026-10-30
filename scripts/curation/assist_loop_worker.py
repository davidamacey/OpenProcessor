#!/usr/bin/env python3
"""Dedicated auto-label loop worker: own container, own logs, survives API restarts.

A loop is a multi-hour job, and splitting a large cluster holds hundreds of MB, so it
does not run in an API process. The API creates the loop's job directory and writes the
project's ``trigger.json`` (see :mod:`src.services.curation.assist_loop.job`); this worker
claims the trigger by unlinking it and runs the loop with its project bound. A resume writes
the same trigger, and the loop replays its event log and continues.

Each poll cycle looks at every active, unpaused project and runs the loop whose trigger is
the oldest, one loop at a time across projects (the VLM is the shared bottleneck). A loop
cancels through its ``cancel.flag`` and pauses through ``pause.flag``; the driver checks both
between requests. At start, a loop a dead worker left ``running`` is repaired to
``interrupted``: it waits for an explicit resume.

Usage:

    python -m scripts.curation.assist_loop_worker
    python -m scripts.curation.assist_loop_worker --project my-project

Or via compose: ``docker compose --profile curation up -d curation-assist-loop-worker``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException


# Make the repo root importable when run as a module from /app/ in the container.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.curation._project_worker_utils import unpaused_projects
from src.config.project_context import bind_project
from src.config.settings import get_settings
from src.services.curation.assist_loop import job
from src.services.curation.assist_loop.runtime import run_job
from src.services.curation.worker_liveness import heartbeat_loop
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import (
    add_project_argument,
    bind_script_project,
    script_project_registry,
)
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(name)s | %(message)s')
logger = logging.getLogger('assist_loop_worker')

POLL_INTERVAL_S = 1.0
WORKER_NAME = 'assist_loop_worker'


def repair_orphans(projects: list[ProjectRecord]) -> None:
    """A loop a dead worker left active becomes ``interrupted`` (resumable by the user)."""
    for record in projects:
        with bind_project(record):
            if job.reconcile_orphaned_jobs():
                logger.warning('project=%s a loop was left running; now interrupted', record.slug)


def _oldest_pending(projects: list[ProjectRecord]) -> ProjectRecord | None:
    best: tuple[float, ProjectRecord] | None = None
    for record in projects:
        with bind_project(record):
            written = job.pending_trigger_mtime()
        if written is not None and (best is None or written < best[0]):
            best = (written, record)
    return best[1] if best else None


async def _run(record: ProjectRecord, loop_id: str, opensearch: Any) -> None:
    logger.info('project=%s loop=%s starting', record.slug, loop_id)
    try:
        outcome = await run_job(opensearch, loop_id)
    except BaseException as exc:
        # A bug, not a refusal (run_job turns those into a failed loop): leave the loop
        # consistent and let the worker crash loudly instead of serving on.
        job.open_job(loop_id).update(
            status='failed',
            error=f'{type(exc).__name__}: {exc}',
            finished_at=now_iso(),
            updated_at=now_iso(),
        )
        raise
    logger.info('project=%s loop=%s %s', record.slug, loop_id, outcome.status)


async def run_next(projects: list[ProjectRecord], opensearch: Any) -> bool:
    """Run the loop with the oldest pending trigger, if any; ``True`` when one ran."""
    record = _oldest_pending(projects)
    if record is None:
        return False
    with bind_project(record):
        loop_id = job.claim_trigger()
        if loop_id is None:
            return False
        await _run(record, loop_id, opensearch)
    return True


async def _main_loop(stop: asyncio.Event, only_slug: str | None) -> None:
    settings = get_settings()
    opensearch = make_script_opensearch(
        [settings.opensearch_url], timeout=settings.opensearch_timeout
    )
    registry = script_project_registry(client=opensearch)
    ticker = asyncio.create_task(heartbeat_loop(WORKER_NAME, lambda: {'poll': True}, stop))
    logger.info('worker ready project=%s', only_slug or '*')
    repaired = False
    try:
        while not stop.is_set():
            try:
                projects = await unpaused_projects(registry, only_slug)
            except (
                OpenSearchException,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:  # retried next cycle
                logger.warning('registry unavailable: %s', exc)
                projects = []
            if not repaired and projects:
                repair_orphans(projects)
                repaired = True
            if await run_next(projects, opensearch):
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_S)
    finally:
        stop.set()
        await ticker
        with contextlib.suppress(OpenSearchException, OSError):
            await opensearch.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_argument(parser)
    parser.set_defaults(project=None)  # every active project; --project restricts to one
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.project:
        bind_script_project(args.project)  # fails fast on an unknown slug
    stop = asyncio.Event()

    async def _amain() -> int:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, stop.set)
        await _main_loop(stop, args.project)
        return 0

    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())

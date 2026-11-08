"""The dispatcher: turns due triggers of registered sources into poll and bulk-run jobs.

:func:`dispatch_once` looks at every source of a project and :func:`decide` s, from the source's
state and queue counts only, what is due:

- ``run_scan``: the cron schedule fired. A run (scan, then drain) is started; fires missed
  while the process was down collapse into one.
- ``poll``: the watch interval elapsed. A ``source_poll`` job scans on its own, without a run,
  and starts a run only if rows are pending afterwards (a quiet bucket makes no runs).
- ``drain``: rows are pending (a webhook, or a run that was interrupted) and a trigger
  is enabled. Held after a cancel, and for five minutes after a failed run.

A scan that fails (the store is unreachable, credentials missing, a prefix removed from
the allow-list) records its error and backs off, doubling to ten minutes. Everything the
dispatcher needs is in the database, so a process that dies loses nothing, and a start that
meets a live job of the same group is skipped: one run per project, one poll per source.

Until the scheduler lane turns the loop into a beat task, :func:`run_tick` calls
:func:`dispatch_once` for every active project on an asyncio interval.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy.exc import SQLAlchemyError

from src.config.env import env_float
from src.config.project_context import current_project_id
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.repos import ingest_sources as repo
from src.db.session import async_session_factory, project_tx
from src.services.curation.bulk_ingest.schedule import (
    Cron,
    InvalidCronError,
    interval_due,
    schedule_due,
)
from src.services.curation.bulk_ingest.source_runs import (
    BulkBusyError,
    last_run,
    run_summary,
    start_poll,
    start_source_run,
)
from src.services.curation.bulk_ingest.sources import IngestSource, from_row


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

DEFAULT_TICK_S = 5
MAX_BACKOFF_S = 600.0
FAILED_RUN_HOLD_S = 300.0
ActionKind = Literal['run_scan', 'poll', 'drain']


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    source_id: str


def tick_interval_s() -> float:
    return env_float('OP_INGEST_DISPATCH_S', float(DEFAULT_TICK_S), minimum=0.0)


def _held(last_run: dict[str, Any] | None, now: float) -> bool:
    """A cancelled run is not restarted by the dispatcher; a failed one waits."""
    if not last_run:
        return False
    if last_run['status'] == 'cancelled':
        return True
    if last_run['status'] == 'failed' and last_run.get('finished_at'):
        finished = datetime.fromisoformat(last_run['finished_at']).timestamp()
        return now - finished < FAILED_RUN_HOLD_S
    return False


def decide(
    source: IngestSource,
    state: dict[str, Any],
    counts: dict[str, int],
    last_run: dict[str, Any] | None,
    *,
    now: float,
) -> Action | None:
    """What, if anything, is due for ``source`` at ``now``. Pure: no I/O."""
    if now < (state.get('backoff_until') or 0.0):
        return None
    if source.schedule.enabled and source.schedule.cron:
        last = state.get('last_fire_at')
        if last:
            try:
                due = schedule_due(
                    Cron.parse(source.schedule.cron),
                    datetime.fromisoformat(last),
                    datetime.fromtimestamp(now, UTC),
                )
            except (InvalidCronError, ValueError) as exc:
                # validated when saved, so a stored one that no longer parses is worth a line
                logger.warning(
                    'ingest_source_schedule_unreadable', source=source.source_id, error=str(exc)
                )
                due = False
            if due:
                return Action('run_scan', source.source_id)
    if source.watch.enabled and interval_due(
        state.get('last_poll_started_at'), source.watch.interval_s, now
    ):
        return Action('poll', source.source_id)
    triggered = source.watch.enabled or source.schedule.enabled or source.webhook_enabled
    if triggered and counts.get('pending', 0) > 0 and not _held(last_run, now):
        return Action('drain', source.source_id)
    return None


def backoff_s(failures: int) -> float:
    """Seconds a failing source waits before its next poll: 10, 20, 40... up to ten minutes."""
    typed: float = min(MAX_BACKOFF_S, 10.0 * 2 ** max(0, failures - 1))
    return typed


async def _start(
    session: AsyncSession, project: uuid.UUID, source: IngestSource, action: Action, moment: float
) -> bool:
    """Start the job of ``action``; ``False`` when a live job of its group makes it wait."""
    source_id = uuid.UUID(source.source_id)
    if action.kind == 'poll':
        if not await start_poll(session, project, source):
            return False
        stamp: dict[str, Any] = {'last_poll_started_at': moment}
    else:
        try:
            await start_source_run(session, project, source, scan=action.kind == 'run_scan')
        except BulkBusyError:
            return False  # another run holds the project; the next tick tries again
        stamp = (
            {'last_fire_at': datetime.fromtimestamp(moment, UTC).isoformat()}
            if action.kind == 'run_scan'
            else {}
        )
    if stamp:
        async with project_tx(session, project):
            await session.run_sync(lambda sync: repo.state_update(sync, source_id, stamp))
    return True


async def dispatch_once(
    session: AsyncSession, project: uuid.UUID, *, now: float | None = None
) -> list[Action]:
    """One decision pass over the project's registered sources; returns the actions it started.

    The one function the tick loop calls today and the periodic task calls later."""
    moment = now if now is not None else time.time()
    async with project_tx(session, project):
        rows = await session.run_sync(lambda sync: repo.list_for_project(sync, project))
    taken: list[Action] = []
    for row in rows:
        source = from_row(row)
        if source.one_shot or not source.triggered:
            continue
        state = dict(row.scan_state)
        if source.schedule.enabled and not state.get('last_fire_at'):
            # A schedule starts counting from the moment it was seen, not from the epoch.
            state['last_fire_at'] = datetime.fromtimestamp(moment, UTC).isoformat()
            async with project_tx(session, project):
                await session.run_sync(
                    repo.state_update, row.id, {'last_fire_at': state['last_fire_at']}
                )
        async with project_tx(session, project):
            counts = await session.run_sync(repo.counts, row.id)
        action = decide(
            source,
            state,
            counts,
            run_summary(await last_run(session, project, state)),
            now=moment,
        )
        if action is not None and await _start(session, project, source, action, moment):
            taken.append(action)
    return taken


async def run_tick() -> None:
    from src.services.projects.bootstrap import for_each_project

    for slug in for_each_project():
        try:
            async with async_session_factory()() as session:
                await dispatch_once(session, current_project_id())
        except (SQLAlchemyError, RequestRejectedError, OSError) as exc:
            # one project's failure must not stop the others' ticks
            logger.warning('ingest_dispatch_failed', project=slug, error=str(exc))


async def _loop(interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s * random.uniform(0.8, 1.2))
        await run_tick()


def start_ingest_dispatcher() -> asyncio.Task[None] | None:
    """The periodic dispatch as a task the caller cancels at shutdown; ``None`` when
    ``OP_INGEST_DISPATCH_S`` is 0."""
    interval = tick_interval_s()
    if interval <= 0:
        return None
    return asyncio.create_task(_loop(interval))

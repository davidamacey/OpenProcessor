"""The dispatcher: turns due triggers of registered sources into scans and bulk runs.

One periodic task per API worker process. Each tick, per active project and under a
non-blocking file lock (so one worker acts per project per tick), it looks at every
source and :func:`decide` s, from the source's state and queue counts only, what is due:

- ``run_scan``: the cron schedule fired. A run (scan, then drain) is started; fires
  missed while the process was down collapse into one.
- ``poll``: the watch interval elapsed. A scan pass runs on its own, without a run, and
  a run is started only if rows are pending afterwards (a quiet bucket makes no runs).
- ``drain``: rows are pending (a webhook, or a run that was interrupted) and a trigger
  is enabled. Held after a cancel, and for five minutes after a failed run.

A scan that fails (the store is unreachable, credentials missing, a prefix removed from
the allow-list) records its error and backs off, doubling to ten minutes. Everything the
dispatcher needs is on disk, so a worker that dies loses nothing: a stale run becomes
``interrupted`` and is resumed by the next ``drain``; a scan resumes from its checkpoint.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from src.config import get_curation_config
from src.config.env import env_float, env_str
from src.core.logging import get_logger
from src.services.curation.bulk_ingest import runner
from src.services.curation.bulk_ingest.schedule import (
    Cron,
    InvalidCronError,
    interval_due,
    schedule_due,
)
from src.services.curation.bulk_ingest.source_runs import last_run_summary, start_source_run
from src.services.curation.bulk_ingest.source_scan import scan_source
from src.services.curation.bulk_ingest.sources import (
    IngestSource,
    list_sources,
    load_source,
    source_queue,
    source_state,
    sources_root,
)
from src.services.curation.bulk_ingest.store import active_run
from src.services.curation.job_lock import exclusive_start_lock


if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

DEFAULT_TICK_S = 5
DEFAULT_BASE_URL = 'http://127.0.0.1:8000'
MAX_BACKOFF_S = 600.0
FAILED_RUN_HOLD_S = 300.0
ActionKind = Literal['run_scan', 'poll', 'drain']
_scans: dict[tuple[str, str], asyncio.Task[None]] = {}
# 'tasks': the loop run and poll tasks belong to (the tick itself runs in a thread).
_loops: dict[str, asyncio.AbstractEventLoop] = {}


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    source_id: str


def tick_interval_s() -> float:
    return env_float('OP_INGEST_DISPATCH_S', float(DEFAULT_TICK_S), minimum=0.0)


def loopback_url() -> str:
    return env_str('OP_API_LOOPBACK_URL') or DEFAULT_BASE_URL


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
            except (InvalidCronError, ValueError):
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


def _backoff(failures: int) -> float:
    typed: float = min(MAX_BACKOFF_S, 10.0 * 2 ** max(0, failures - 1))
    return typed


async def _poll(source: IngestSource, *, start_run: Callable[[IngestSource, bool], object]) -> None:
    """One watcher pass: scan into the queue, then start a drain if rows are pending."""
    state = source_state(source.source_id)

    def stop() -> bool:
        return load_source(source.source_id) is None

    try:
        await asyncio.to_thread(scan_source, source, should_stop=stop)
    except Exception as exc:
        failures = int(state.read().get('failures') or 0) + 1
        state.update(
            failures=failures,
            last_poll_error=f'{type(exc).__name__}: {exc}'[:300],
            backoff_until=time.time() + _backoff(failures),
        )
        logger.warning('ingest_source_poll_failed', source=source.source_id, error=str(exc))
        return
    state.update(failures=0, last_poll_error=None, backoff_until=None)
    if source_queue(source.source_id).counts()['pending'] > 0:
        start_run(source, False)


def _start(source: IngestSource, scan: bool) -> None:
    try:
        start_source_run(
            source,
            base_url=loopback_url(),
            api_prefix=get_curation_config().api_prefix,
            project=get_curation_config().project_slug,
            scan=scan,
            loop=_loops.get('tasks'),
        )
    except runner.BulkBusyError:
        return  # another run holds the project; the next tick tries again
    if scan:
        source_state(source.source_id).update(last_fire_at=datetime.now(UTC).isoformat())


def _on_task_loop() -> bool:
    try:
        return asyncio.get_running_loop() is _loops.get('tasks')
    except RuntimeError:
        return False


def _execute(action: Action, source: IngestSource) -> None:
    key = (get_curation_config().project_slug, source.source_id)
    if action.kind == 'poll':
        running = _scans.get(key)
        if running is not None and not running.done():
            return
        # Stamped here, under the tick's lock, so a second worker's tick sees it at once.
        source_state(source.source_id).update(last_poll_started_at=time.time())

        def begin() -> None:
            task = asyncio.create_task(_poll(source, start_run=_start))
            _scans[key] = task
            task.add_done_callback(lambda _t: _scans.pop(key, None))

        task_loop = _loops.get('tasks')
        if task_loop is not None and task_loop.is_running() and not _on_task_loop():
            task_loop.call_soon_threadsafe(begin)
        else:
            begin()
    elif action.kind == 'run_scan':
        _start(source, True)
    else:
        _start(source, False)


def run_project_tick(*, now: float | None = None) -> list[Action]:
    """One tick for the bound project; returns the actions it took."""
    moment = now if now is not None else time.time()
    taken: list[Action] = []
    root = sources_root()
    if not root.is_dir():
        return taken
    with exclusive_start_lock(root / 'dispatch.lock') as mine:
        if not mine:
            return taken
        live = active_run() is not None
        for source in list_sources():
            state_file = source_state(source.source_id)
            state = state_file.read()
            if source.schedule.enabled and not state.get('last_fire_at'):
                # A schedule starts counting from the moment it was seen, not from the epoch.
                state_file.update(last_fire_at=datetime.fromtimestamp(moment, UTC).isoformat())
                state = state_file.read()
            action = decide(
                source,
                state,
                source_queue(source.source_id).counts(),
                last_run_summary(state),
                now=moment,
            )
            if action is None or (action.kind != 'poll' and live):
                continue
            _execute(action, source)
            taken.append(action)
            if action.kind != 'poll':
                live = True  # one run per project: the rest wait for the next tick
    return taken


async def run_tick() -> None:
    from src.services.projects.bootstrap import for_each_project

    _loops['tasks'] = asyncio.get_running_loop()
    for slug in for_each_project():
        try:
            # The tick does blocking file I/O, so it runs in a thread; the run and the poll
            # are asyncio tasks, created on `_loops['tasks']`.
            await asyncio.to_thread(run_project_tick)
        except Exception as exc:  # one project's failure must not stop the others' ticks
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

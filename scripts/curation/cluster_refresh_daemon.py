#!/usr/bin/env python3
"""Cluster refresh daemon — Task #92 Part C.

Polls each project's items index and, once the count has grown by a
configurable threshold since the last retrain, runs ``auto_promote`` and
queues the full cluster retrain as the background ``auto_label`` job (it
runs in the auto-label worker, never inside an API process). The retrain
waits for ingest to go quiet (item count flat since the previous poll and
no region work unfinished) for at most ``--max-deferral-seconds``; repeated
requests coalesce into one, and a project never has two retrains in flight.
Pausing a project cancels the job this daemon started for it. The schedule
is ``src/services/curation/clustering/refresh_schedule.py``; the work is
unchanged.

Defaults:

* poll interval: 120 seconds (2 minutes)
* growth threshold: 200 new crops
* maximum deferral: 1800 seconds
* incremental auto-label: on (turn off with ``--no-auto-label``)

Designed to run as a long-lived process (``python -m
scripts.curation.cluster_refresh_daemon``) or as a cron job invoking
``--once``.

Projects: each iteration discovers the active projects, skips paused
ones (``<project_state_dir>/pipeline_paused.flag``) and checks every
other one in rotating order, with a growth counter per project. A
project's count reads its own items index (guarded client, bound to that
project); its refresh calls its own ``{prefix}/projects/{slug}/...``
routes. ``--project SLUG`` restricts the daemon to one project.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
import time
from pathlib import Path
from typing import Any

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.curation._project_worker_utils import (
    curation_api_prefix,
    rotated,
    scoped_url,
    unpaused_projects,
)
from src.config.project_context import bind_project
from src.services.curation.clustering.refresh_schedule import (
    ACTIVE_JOB_STATUSES,
    Action,
    Observation,
    Policy,
    ProjectSchedule,
    ScheduleBook,
    step,
)
from src.services.curation.worker_liveness import write_heartbeat
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import (
    add_project_argument,
    bind_script_project,
    script_project_registry,
)


# S-2: container healthcheck liveness. The daemon's real poll interval
# (default 5 min) is far longer than any sane healthcheck max-age, so
# the inter-iteration wait is sliced into chunks of this size and the
# heartbeat is touched once per chunk — see run().
_LIVENESS_TICK_S = 15.0


DEFAULT_API = 'http://localhost:4603'
DEFAULT_OS = 'http://localhost:4607'


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api', default=DEFAULT_API, help='triton-api base URL')
    p.add_argument('--opensearch', default=DEFAULT_OS, help='OpenSearch base URL')
    p.add_argument(
        '--interval-seconds',
        type=int,
        default=120,
        help='Poll interval (default 120 = 2 min).',
    )
    p.add_argument(
        '--growth-threshold',
        type=int,
        default=200,
        help='Trigger refresh once the items index grows by this many docs since the last refresh.',
    )
    p.add_argument(
        '--max-deferral-seconds',
        type=int,
        default=1800,
        help='Longest a due retrain waits for ingest to go quiet before it starts anyway.',
    )
    p.add_argument(
        '--retry-backoff-seconds',
        type=int,
        default=600,
        help='Wait before retrying a retrain job that failed, was cancelled or was lost.',
    )
    p.add_argument(
        '--auto-label',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Queue the auto_label background job (the full retrain) after auto_promote so freshly-'
        'CNN-labeled crops get assigned to their named class clusters '
        '(force_cluster_id_equals_class_id) and unlabeled residuals are '
        'fed through the VLM/segmenter. Default on; pass --no-auto-label to '
        'skip.',
    )
    p.add_argument(
        '--once',
        action='store_true',
        help='Run a single iteration and exit (cron-friendly).',
    )
    add_project_argument(p)
    p.set_defaults(project=None)  # every active project; --project restricts to one
    return p.parse_args()


async def _crop_count(opensearch: Any) -> int:
    """The bound project's item count."""
    from src.config.curation import items_index

    resp = await opensearch.count(index=items_index())
    return int(resp.get('count', 0))


async def _unfinished_region_count(opensearch: Any) -> int:
    """Items still waiting for the region stage (the ``region_drain`` facts)."""
    from src.config import get_region_fields
    from src.config.curation import items_index
    from src.config.region_state import RegionStatus

    resp = await opensearch.count(
        index=items_index(),
        body={
            'query': {
                'terms': {
                    get_region_fields().status: [
                        RegionStatus.PENDING_DETECTION,
                        RegionStatus.PENDING_VERIFICATION,
                    ]
                }
            }
        },
    )
    return int(resp.get('count', 0))


async def _trigger_auto_promote(
    client: httpx.AsyncClient, api: str, api_prefix: str, slug: str
) -> dict[str, Any]:
    r = await client.post(
        scoped_url(api, api_prefix, slug, '/clusters/auto_promote'),
        json={},
        # A cold-start pass (daemon restart resets in-process last_count to
        # 0, so the very next poll always re-triggers over the *full* pool,
        # not just recent growth) synchronously OCC-checks/writes every
        # crop and has been observed taking >600s over a 347k-doc pool --
        # 600s wasn't enough, causing a perpetual timeout/retry loop.
        timeout=900.0,
    )
    r.raise_for_status()
    return r.json()


async def _start_auto_label_job(
    client: httpx.AsyncClient, api: str, api_prefix: str, slug: str
) -> str | None:
    """Queue the retrain as the background auto-label job (it runs in the
    auto-label worker, not in an API process). No query: the job route's
    defaults are the synchronous route's defaults. ``None`` when a job is
    already in flight (409): the request stays pending."""
    r = await client.post(scoped_url(api, api_prefix, slug, '/pipeline/auto_label/start'), json={})
    if r.status_code == httpx.codes.CONFLICT:
        return None
    r.raise_for_status()
    return str(r.json().get('job_id') or '') or None


async def _job_status(
    client: httpx.AsyncClient, api: str, api_prefix: str, slug: str, job_id: str | None = None
) -> dict[str, Any]:
    """The project's current auto-label state, or one job's by id. The route
    repairs a run whose worker died (stale heartbeat) to ``interrupted``."""
    path = '/pipeline/auto_label/status' + (f'/{job_id}' if job_id else '')
    r = await client.get(scoped_url(api, api_prefix, slug, path))
    if r.status_code == httpx.codes.NOT_FOUND:
        return {'status': 'unknown'}
    r.raise_for_status()
    return r.json()


async def _cancel_job(client: httpx.AsyncClient, api: str, api_prefix: str, slug: str) -> None:
    r = await client.post(scoped_url(api, api_prefix, slug, '/pipeline/auto_label/cancel'))
    r.raise_for_status()


async def _release_inactive(
    client: httpx.AsyncClient,
    *,
    api: str,
    api_prefix: str,
    book: ScheduleBook,
    active: set[str],
    paused: set[str],
) -> None:
    """Projects that left the active set: a paused one has its retrain
    cancelled (the cooperative cancel flag) and the request re-raised on
    resume; a deleted one (the delete removes its job directory) is dropped."""
    for slug in book.slugs():
        if slug in active:
            continue
        sched = book.get(slug)
        if slug in paused and sched.job_id is not None:
            try:
                await _cancel_job(client, api, api_prefix, slug)
                print(f'[cluster-refresh] project={slug} paused: retrain cancelled', flush=True)
            except httpx.HTTPError as exc:
                print(f'[cluster-refresh] project={slug} cancel failed: {exc}', flush=True)
            sched.on_cancelled()
        elif slug not in paused:
            book.forget(slug)


async def _iteration(
    client: httpx.AsyncClient,
    *,
    api: str,
    api_prefix: str,
    record: Any,
    opensearch: Any,
    sched: ProjectSchedule,
    policy: Policy,
    auto_label: bool,
    now: float,
) -> None:
    """One poll + (maybe) refresh of ``record``."""
    slug = record.slug
    try:
        with bind_project(record):
            count = await _crop_count(opensearch)
            unfinished = await _unfinished_region_count(opensearch)
        busy = False
        tracked = None
        if auto_label:
            busy = (await _job_status(client, api, api_prefix, slug)).get(
                'status'
            ) in ACTIVE_JOB_STATUSES
            if sched.job_id is not None:
                tracked = (await _job_status(client, api, api_prefix, slug, sched.job_id)).get(
                    'status'
                )
    except Exception as exc:
        print(f'[cluster-refresh] project={slug} observation failed: {exc}', flush=True)
        return
    action = step(
        sched,
        Observation(count=count, unfinished=unfinished, tracked_status=tracked, busy=busy),
        now,
        policy,
    )
    print(
        f'[cluster-refresh] project={slug} crops={count} unfinished_regions={unfinished} '
        f'trained_at={sched.last_trained_count} threshold={policy.threshold} action={action.value}',
        flush=True,
    )
    if action is not Action.START:
        return
    print(f'[cluster-refresh] project={slug} triggering auto_promote', flush=True)
    try:
        promo = await _trigger_auto_promote(client, api, api_prefix, slug)
        print(f'[cluster-refresh] project={slug} auto_promote result: {promo}', flush=True)
    except httpx.HTTPError as exc:
        print(f'[cluster-refresh] project={slug} auto_promote failed: {exc}', flush=True)
        return
    previous = sched.last_trained_count
    if not auto_label:
        sched.on_started(count, None, previous_trained=previous)
        return
    try:
        job_id = await _start_auto_label_job(client, api, api_prefix, slug)
    except httpx.HTTPError as exc:
        print(f'[cluster-refresh] project={slug} auto_label start failed: {exc}', flush=True)
        return
    if job_id is None:
        print(
            f'[cluster-refresh] project={slug} auto_label already running; kept pending', flush=True
        )
        return
    sched.on_started(count, job_id, previous_trained=previous)
    print(f'[cluster-refresh] project={slug} auto_label job {job_id} queued', flush=True)


def _make_signal_stop() -> tuple[asyncio.Event, None]:
    stop = asyncio.Event()

    def _on_signal(*_: object) -> None:
        if not stop.is_set():
            print('[cluster-refresh] stop requested', flush=True)
            stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)
    return stop, None


async def _sleep_with_heartbeat(stop: asyncio.Event, sleep_s: float) -> None:
    # Slice the inter-iteration wait into <=_LIVENESS_TICK_S chunks so
    # the container heartbeat stays fresh across a multi-minute wait.
    remaining = sleep_s
    while remaining > 0 and not stop.is_set():
        chunk = min(remaining, _LIVENESS_TICK_S)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=chunk)
        remaining -= chunk
        write_heartbeat('cluster_refresh', {'loop': True})


async def run(args: argparse.Namespace) -> int:
    stop, _ = _make_signal_stop()
    registry = script_project_registry(args.opensearch)
    opensearch = make_script_opensearch([args.opensearch], timeout=30)
    api_prefix = curation_api_prefix()
    book = ScheduleBook()
    policy = Policy(
        threshold=args.growth_threshold,
        max_deferral_s=args.max_deferral_seconds,
        retry_backoff_s=args.retry_backoff_seconds,
    )
    rotation = 0
    write_heartbeat('cluster_refresh', {'loop': True})
    async with httpx.AsyncClient(timeout=30.0) as client:
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                everyone = await unpaused_projects(registry, None)
                projects = rotated(
                    [r for r in everyone if args.project in (None, r.slug)], rotation
                )
                active = {r.slug for r in everyone}
                paused = {
                    r.slug
                    for r in registry.active_projects()
                    if r.slug not in active and args.project in (None, r.slug)
                }
            except Exception as exc:
                print(f'[cluster-refresh] registry unavailable: {exc}', flush=True)
                projects, active, paused = [], set(book.slugs()), set()
            rotation += 1
            await _release_inactive(
                client, api=args.api, api_prefix=api_prefix, book=book, active=active, paused=paused
            )
            for record in projects:
                await _iteration(
                    client,
                    api=args.api,
                    api_prefix=api_prefix,
                    record=record,
                    opensearch=opensearch,
                    sched=book.get(record.slug),
                    policy=policy,
                    auto_label=args.auto_label,
                    now=time.monotonic(),
                )
                write_heartbeat('cluster_refresh', {'loop': True})
            write_heartbeat('cluster_refresh', {'loop': True})
            if args.once:
                break
            await _sleep_with_heartbeat(
                stop, max(args.interval_seconds - (time.monotonic() - t0), 1.0)
            )
    await opensearch.close()
    return 0


def main() -> int:
    args = _parse_args()
    if args.project:
        # Fails fast on an unknown/unbindable slug.
        bind_script_project(args.project, opensearch_url=args.opensearch)
    return asyncio.run(run(args))


if __name__ == '__main__':
    sys.exit(main())

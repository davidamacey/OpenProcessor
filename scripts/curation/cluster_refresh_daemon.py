#!/usr/bin/env python3
"""Cluster refresh daemon — Task #92 Part C.

Polls the curation items index total every N minutes and triggers
``/curation/clusters/auto_promote`` (and optionally ``/curation/pipeline/auto_label``
on the new crops) once the count grows by a configurable threshold
since the last refresh. The result is that periodic clustering happens
"organically" while ingest is running, so the labeler /clusters page
keeps surfacing fresh clusters without the user remembering to kick
the pipeline by hand.

Defaults match the design discussion in Task #92:

* poll interval: 120 seconds (2 minutes)
* growth threshold: 1000 new crops
* incremental auto-label: off (turn on with ``--auto-label``)

Designed to run as a long-lived process (``python -m
scripts.curation.cluster_refresh_daemon``) or as a cron job invoking
``--once``.

Multi-project mode (default): with no ``--project``, each iteration
discovers every active project and refreshes one of them (rotating one
per iteration), skipping a project whose ``pipeline_paused.flag`` is
set. ``--project SLUG`` restricts to (and whole-process-binds) one
project -- the original single-project growth-tracking loop, unchanged.
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
    project_api_prefix,
    project_items_index,
    project_paused,
    scoped_url,
)
from src.services.curation.worker_liveness import write_heartbeat
from src.services.projects.script_binding import add_project_argument, bind_script_project


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
        '--auto-label',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='POST /curation/pipeline/auto_label after auto_promote so freshly-'
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
    p.set_defaults(project=None)  # multi-project mode is the new default
    return p.parse_args()


async def _crop_count(client: httpx.AsyncClient, opensearch: str, items_index: str) -> int:
    r = await client.get(f'{opensearch}/{items_index}/_count', timeout=10.0)
    r.raise_for_status()
    return int(r.json().get('count', 0))


async def _trigger_auto_promote(
    client: httpx.AsyncClient, api: str, api_prefix: str, slug: str
) -> dict[str, Any]:
    r = await client.post(
        scoped_url(api, api_prefix, slug, '/clusters/auto_promote'),
        json={},
        # A cold-start pass (daemon restart resets in-process last_count to
        # 0, so the very next poll always re-triggers over the *full* pool,
        # not just recent growth) synchronously OCC-checks/writes every
        # crop and has been observed taking >600s over a 347k-doc pool —
        # 600s wasn't enough, causing a perpetual timeout/retry loop.
        # Matches the 900s timeout already used for the equally-slow
        # /pipeline/auto_label full-pool call.
        timeout=900.0,
    )
    r.raise_for_status()
    return r.json()


async def _trigger_auto_label(
    client: httpx.AsyncClient, api: str, api_prefix: str, slug: str
) -> dict[str, Any]:
    r = await client.post(
        scoped_url(api, api_prefix, slug, '/pipeline/auto_label'),
        json={},
        timeout=900.0,
    )
    r.raise_for_status()
    return r.json()


async def _iteration(
    client: httpx.AsyncClient,
    *,
    api: str,
    api_prefix: str,
    slug: str,
    opensearch: str,
    items_index: str,
    last_count: int,
    threshold: int,
    auto_label: bool,
) -> int:
    """One poll + (maybe) refresh. Returns the new ``last_count``."""
    try:
        count = await _crop_count(client, opensearch, items_index)
    except httpx.HTTPError as exc:
        print(f'[cluster-refresh] project={slug} crop_count failed: {exc}', flush=True)
        return last_count
    growth = count - last_count
    print(
        f'[cluster-refresh] project={slug} crops={count} growth_since_last={growth} '
        f'threshold={threshold}',
        flush=True,
    )
    if last_count > 0 and growth < threshold:
        return last_count
    # First iteration (last_count == 0) always triggers — we want a
    # fresh promote on daemon startup so the user sees the current
    # state reflected in /clusters.
    print(f'[cluster-refresh] project={slug} triggering auto_promote', flush=True)
    try:
        promo = await _trigger_auto_promote(client, api, api_prefix, slug)
        print(f'[cluster-refresh] project={slug} auto_promote result: {promo}', flush=True)
    except httpx.HTTPError as exc:
        print(f'[cluster-refresh] project={slug} auto_promote failed: {exc}', flush=True)
        return last_count
    if auto_label:
        print(f'[cluster-refresh] project={slug} triggering auto_label', flush=True)
        try:
            lab = await _trigger_auto_label(client, api, api_prefix, slug)
            print(f'[cluster-refresh] project={slug} auto_label result: {lab}', flush=True)
        except httpx.HTTPError as exc:
            print(f'[cluster-refresh] project={slug} auto_label failed: {exc}', flush=True)
    return count


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
    # Single, --project-bound project: the original growth-tracking loop.
    from src.config.project_context import current_project

    bound = current_project()
    items_index = project_items_index(bound.record)
    api_prefix = project_api_prefix()
    slug = bound.record.slug

    stop, _ = _make_signal_stop()
    last_count = 0
    write_heartbeat('cluster_refresh', {'loop': True})
    async with httpx.AsyncClient(timeout=30.0) as client:
        while not stop.is_set():
            t0 = time.monotonic()
            last_count = await _iteration(
                client,
                api=args.api,
                api_prefix=api_prefix,
                slug=slug,
                opensearch=args.opensearch,
                items_index=items_index,
                last_count=last_count,
                threshold=args.growth_threshold,
                auto_label=args.auto_label,
            )
            write_heartbeat('cluster_refresh', {'loop': True})
            if args.once:
                break
            await _sleep_with_heartbeat(
                stop, max(args.interval_seconds - (time.monotonic() - t0), 1.0)
            )
    return 0


async def run_multi_project(args: argparse.Namespace) -> int:
    # Round-robins every active project, one at a time per iteration.
    from src.services.projects.registry import get_project_registry

    stop, _ = _make_signal_stop()
    registry = get_project_registry()
    api_prefix = project_api_prefix()
    last_counts: dict[str, int] = {}
    rotation = 0
    write_heartbeat('cluster_refresh', {'loop': True})
    async with httpx.AsyncClient(timeout=30.0) as client:
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                await registry.ensure_fresh()
                active = registry.active_projects()
            except Exception as exc:
                print(f'[cluster-refresh] registry unavailable: {exc}', flush=True)
                active = []

            if active:
                rotation %= len(active)
                record = active[rotation]
                rotation += 1
                try:
                    paused = project_paused(record)
                    items_index = project_items_index(record) if not paused else ''
                except Exception as exc:
                    print(f'[cluster-refresh] project {record.slug} unavailable: {exc}', flush=True)
                    paused, items_index = True, ''
                if paused:
                    print(f'[cluster-refresh] project={record.slug} paused, skipping', flush=True)
                else:
                    last_counts[record.slug] = await _iteration(
                        client,
                        api=args.api,
                        api_prefix=api_prefix,
                        slug=record.slug,
                        opensearch=args.opensearch,
                        items_index=items_index,
                        last_count=last_counts.get(record.slug, 0),
                        threshold=args.growth_threshold,
                        auto_label=args.auto_label,
                    )

            write_heartbeat('cluster_refresh', {'loop': True})
            if args.once:
                break
            await _sleep_with_heartbeat(
                stop, max(args.interval_seconds - (time.monotonic() - t0), 1.0)
            )
    return 0


def main() -> int:
    args = _parse_args()
    if args.project:
        bind_script_project(args.project, opensearch_url=args.opensearch)
        return asyncio.run(run(args))
    return asyncio.run(run_multi_project(args))


if __name__ == '__main__':
    sys.exit(main())

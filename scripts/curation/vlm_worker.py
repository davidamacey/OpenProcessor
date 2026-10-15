#!/usr/bin/env python3
"""Async VLM labeling worker — overlaps the VLM (GPU 2) with ingest (GPU 0).

Background loop that polls OpenSearch for crops that need the VLM's vision
verdict and dispatches them to /curation/vlm/label_batch in parallel with
ongoing image ingestion. This unsticks the pipeline at HDD scale where
the previous "ingest everything → then VLM" sequence wasted 50%+ of
elapsed time waiting for one GPU while the other was idle.

Selection matches ``pipeline_auto_label``'s skip logic. Idempotent
(a successful VLM write drops the crop from the next poll's query),
auto-exits after ``--idle-stop-after`` empty polls, and is stoppable
with SIGINT/SIGTERM (finishes the in-flight batch first).

Usage
-----
    .venv/bin/python scripts/curation/vlm_worker.py
    .venv/bin/python scripts/curation/vlm_worker.py --batch-size 32 --concurrency 4
    .venv/bin/python scripts/curation/vlm_worker.py --until-empty   # one drain pass

Projects: each producer cycle discovers the active projects, skips
paused ones (``<project_state_dir>/pipeline_paused.flag``) and gives
each a fair share of the fetch, rotating which project goes first. A
project's crops are read from its own items index (through the guarded
OpenSearch client, bound to that project) and sent to its own
``{prefix}/projects/{slug}/vlm/label_batch``. ``--project SLUG``
restricts the worker to one project.

Or as the long-lived compose service (G-10: there is no
``make curation-vlm-worker`` target -- use one of these instead):
    docker compose --profile curation up -d curation-vlm-worker
    docker compose exec api python scripts/curation/vlm_worker.py --until-empty
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from scripts.curation._project_worker_utils import curation_api_prefix, rotated, scoped_url
from scripts.curation._vlm_activity import idle_backoff_s, projects_with_active_vlm
from scripts.curation.vlm_cli import parse_args
from scripts.curation.vlm_pending import _filter_fresh_ids, fetch_pending_ids
from src.config.project_context import bind_project
from src.services.curation.ops_metrics import start_worker_metrics_server
from src.services.curation.vlm_scope import ScopeCache
from src.services.curation.worker_liveness import heartbeat_loop
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import bind_script_project, script_project_registry


if TYPE_CHECKING:
    import argparse

# How long a released-then-not-yet-refreshed crop id stays in the
# released_at guard. Mirrors scripts/curation/worker/runner.py's
# _RELEASED_AT_TTL_S.
_RELEASED_AT_TTL_S = 300.0

# Backoff after the VLM endpoint (or the API) stops answering: starts at the poll
# interval, doubles per consecutive failed probe, capped here.
_OUTAGE_BACKOFF_MAX_S = 60.0


def _error_code(response: httpx.Response) -> str | None:
    try:
        detail = response.json().get('detail')
    except ValueError:
        return None
    return detail.get('error') if isinstance(detail, dict) else None


def _error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get('detail')
    except ValueError:
        return ''
    return str(detail.get('message', '')) if isinstance(detail, dict) else ''


async def label_batch(
    client: httpx.AsyncClient,
    *,
    api: str,
    api_prefix: str,
    slug: str,
    crop_ids: list[str],
) -> dict:
    """Call the scoped /curation/projects/{slug}/vlm/label_batch for one chunk.

    Returns ``{'no_classes': True}`` when the project has no classes yet, and
    ``{'vlm_unreachable': True, 'message': ...}`` when the VLM endpoint did not
    answer, so the caller backs off instead of logging a failure per chunk."""
    r = await client.post(
        scoped_url(api, api_prefix, slug, '/vlm/label_batch'),
        json={'crop_ids': crop_ids},
        timeout=300.0,
    )
    if r.status_code == 409 and _error_code(r) == 'no_classes':
        return {'no_classes': True}
    if r.status_code == 503 and _error_code(r) == 'vlm_unreachable':
        return {'vlm_unreachable': True, 'message': _error_message(r)}
    r.raise_for_status()
    result: dict = r.json()
    return result


class _StreamWorker:
    """The state one ``run`` shares between its producer, consumers and reporters."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.api_prefix = curation_api_prefix()
        self.opensearch = make_script_opensearch([args.opensearch], timeout=30)
        self.registry = script_project_registry(client=self.opensearch)
        self.scope_cache = ScopeCache()
        self.rotation = 0
        self.stop_event = asyncio.Event()
        self.started_at = time.monotonic()
        self.metrics = {
            'total_processed': 0,
            'total_updated': 0,
            'total_chunks': 0,
            'consecutive_empty_polls': 0,
        }
        # Crop ids currently being processed by a consumer (or queued for
        # a consumer). The producer skips these on its next OS fetch so
        # we don't double-dispatch. Removed by the consumer once it has
        # written the terminal class_source back to OS.
        # Keyed by (project slug, crop id): the same image in two projects has
        # the same crop id.
        self.in_flight: set[tuple[str, str]] = set()
        self.in_flight_lock = asyncio.Lock()
        # label_batch writes with refresh=False, so a producer fetch
        # that starts right after a consumer discards a crop from in_flight
        # can still see the pre-write state and re-dispatch it (duplicate GPU
        # work). Ported from the region/SAM worker's runner.py pattern: hold
        # each released id here for one refresh interval past
        # _RELEASED_AT_TTL_S, keyed by release time, and require a fetch to
        # have started after that release to treat the id as fresh again.
        self.released_at: dict[tuple[str, str], float] = {}
        # Why the worker is making no progress (the VLM endpoint or the API does
        # not answer), or None. While ``backing_off`` the producer sleeps and the
        # consumers release their chunks untouched; when the sleep ends the next
        # chunk is the probe. Crops are never written as failed for an outage.
        self.stall_reason: str | None = None
        self.backing_off = False
        self.outage_backoff_s = 0.0
        # Queue depth: small buffer between producer and consumers. Just
        # big enough to absorb one OS fetch latency. Larger doesn't help
        # — vLLM's max-num-seqs caps real throughput downstream.
        self.queue: asyncio.Queue[tuple[Any, list[str]] | None] = asyncio.Queue(
            maxsize=args.concurrency * 2
        )

    def _on_signal(self, *_: object) -> None:
        if not self.stop_event.is_set():
            print('[vlm-worker] stop requested — finishing in-flight chunks...')
            self.stop_event.set()

    async def _fetch_project(self, record: Any, n: int) -> list[str]:
        """This project's fresh pending ids, read under its own binding."""
        fetch_started = time.monotonic()
        async with self.in_flight_lock:
            exclude_ids = [cid for slug, cid in self.in_flight if slug == record.slug]
        with bind_project(record):
            ids = await fetch_pending_ids(
                self.opensearch,
                slug=record.slug,
                scope_cache=self.scope_cache,
                batch_size=n,
                classifier_skip_conf=self.args.classifier_conf_skip,
                exclude_ids=exclude_ids,
            )
        # Drop ids still in flight, and ids released since (or shortly
        # before) this fetch started -- the write used refresh=False, so
        # a fetch that began around the release may still see stale state.
        async with self.in_flight_lock:
            fresh = _filter_fresh_ids(
                [(record.slug, cid) for cid in ids],
                in_flight=self.in_flight,
                released_at=self.released_at,
                fetch_started=fetch_started,
            )
            horizon = fetch_started - _RELEASED_AT_TTL_S
            for key in [k for k, ts in self.released_at.items() if ts < horizon]:
                del self.released_at[key]
        return [cid for _, cid in fresh]

    async def _enqueue_fresh(self, record: Any, fresh: list[str]) -> None:
        # Chunk + queue. Mark in-flight before queueing so a fast
        # consumer can't race a slow OS write.
        for i in range(0, len(fresh), self.args.vlm_batch_size):
            chunk = fresh[i : i + self.args.vlm_batch_size]
            async with self.in_flight_lock:
                self.in_flight.update((record.slug, cid) for cid in chunk)
            await self.queue.put((record, chunk))

    async def _fetch_cycle(self, projects: list[Any], share: int) -> bool:
        """Fetch and enqueue each project's pending ids; True when any came."""
        fetched_any = False
        for record in projects:
            try:
                fresh = await self._fetch_project(record, share)
            except Exception as exc:
                print(f'[vlm-worker] project={record.slug} fetch error: {exc}')
                continue
            if not fresh:
                continue
            fetched_any = True
            await self._enqueue_fresh(record, fresh)
        return fetched_any

    async def _idle_cycle(self, projects: list[Any]) -> bool:
        """An empty poll: stop the producer (True) once the idle limit is reached,
        otherwise sleep before the next poll."""
        args = self.args
        self.metrics['consecutive_empty_polls'] += 1
        if not args.continuous and (
            args.until_empty or self.metrics['consecutive_empty_polls'] >= args.idle_stop_after
        ):
            # Drain: signal consumers to stop after the queue empties.
            print(
                f'[vlm-worker] producer idle '
                f'({self.metrics["consecutive_empty_polls"]} empty polls) — draining'
            )
            self.stop_event.set()
            return True
        await asyncio.sleep(
            args.poll_interval
            if projects
            else idle_backoff_s(args.poll_interval, self.metrics['consecutive_empty_polls'])
        )
        return False

    def _enter_outage(self, reason: str) -> None:
        """A chunk found the endpoint down: stall, and back off once for all consumers."""
        if self.backing_off:
            return
        self.outage_backoff_s = min(
            max(self.args.poll_interval, self.outage_backoff_s * 2), _OUTAGE_BACKOFF_MAX_S
        )
        self.backing_off = True
        if reason != self.stall_reason:
            print(f'[vlm-worker] {reason}; backing off, crops stay pending')
        self.stall_reason = reason

    def _leave_outage(self) -> None:
        if self.stall_reason is not None:
            print('[vlm-worker] VLM endpoint recovered; resuming')
        self.stall_reason = None
        self.backing_off = False
        self.outage_backoff_s = 0.0

    async def _outage_wait(self) -> None:
        await asyncio.sleep(self.outage_backoff_s)
        self.backing_off = False

    async def producer(self) -> None:
        """Continuously fetch eligible crops, project by project, and chunk
        them into the queue."""
        args = self.args
        while not self.stop_event.is_set():
            # Pause if the GPU arbiter says so (training claimed the GPU).
            if args.pause_sentinel and Path(args.pause_sentinel).exists():
                await asyncio.sleep(args.pause_poll_interval)
                continue
            if self.backing_off:
                await self._outage_wait()
                continue
            # Backpressure: don't outpace the consumers.
            if self.queue.full():
                await asyncio.sleep(0.05)
                continue
            projects = rotated(
                await projects_with_active_vlm(self.opensearch, self.registry, args.project),
                self.rotation,
            )
            self.rotation += 1
            # In-flight ids are excluded server-side (must_not ids), so a
            # fetch only needs to refill the queue.
            fetch_n = min(args.vlm_batch_size * args.concurrency * 2, 9000)  # OS hits cap
            share = max(1, fetch_n // max(1, len(projects)))
            if not await self._fetch_cycle(projects, share):
                if await self._idle_cycle(projects):
                    return
                continue
            self.metrics['consecutive_empty_polls'] = 0

    def _count_chunk(self, consumer_id: int, chunk: list[str], result: dict, t0: float) -> None:
        metrics = self.metrics
        metrics['no_classes_logged'] = 0
        metrics['total_processed'] += int(result.get('predicted', 0))
        metrics['total_updated'] += int(result.get('updated', 0))
        metrics['total_chunks'] += 1
        rate = len(chunk) / max(time.monotonic() - t0, 1e-6)
        # Throttle the per-chunk log to 1-in-10 to keep logs readable.
        if metrics['total_chunks'] % 10 == 0:
            elapsed_total = time.monotonic() - self.started_at
            rate_avg = metrics['total_processed'] / max(elapsed_total, 1e-6)
            print(
                f'[vlm-worker] chunk {metrics["total_chunks"]} '
                f'(consumer {consumer_id}): {len(chunk)} crops in '
                f'{time.monotonic() - t0:.1f}s ({rate:.1f} cps)  '
                f'session={metrics["total_processed"]} '
                f'avg={rate_avg:.1f} cps'
            )

    async def _label_chunk(
        self, consumer_id: int, client: httpx.AsyncClient, record: Any, chunk: list[str]
    ) -> None:
        t0 = time.monotonic()
        try:
            if self.backing_off:
                return  # released untouched below; the producer's probe comes after the backoff
            result = await label_batch(
                client,
                api=self.args.api,
                api_prefix=self.api_prefix,
                slug=record.slug,
                crop_ids=chunk,
            )
            if result.get('vlm_unreachable'):
                message = result.get('message') or 'no answer'
                self._enter_outage(f'VLM endpoint unreachable: {message}')
                return
            self._leave_outage()
            if result.get('no_classes'):
                if not self.metrics.get('no_classes_logged'):
                    print('[vlm-worker] project has no classes yet; idling until some exist')
                    self.metrics['no_classes_logged'] = 1
                await asyncio.sleep(self.args.poll_interval)
                return
            self._count_chunk(consumer_id, chunk, result, t0)
        except httpx.HTTPError as exc:
            print(f'[vlm-worker] consumer {consumer_id} HTTP error: {exc}')
            if isinstance(exc, httpx.TransportError):
                self._enter_outage(f'curation API unreachable: {exc}')
        except Exception as exc:
            print(f'[vlm-worker] consumer {consumer_id} error: {exc}')
        finally:
            released = time.monotonic()
            async with self.in_flight_lock:
                for cid in chunk:
                    self.in_flight.discard((record.slug, cid))
                    self.released_at[(record.slug, cid)] = released
            self.queue.task_done()

    async def consumer(self, consumer_id: int, client: httpx.AsyncClient) -> None:
        """Pull a chunk from the queue, call its project's label_batch, repeat."""
        while True:
            item = await self.queue.get()
            if item is None:  # poison pill = drain complete
                self.queue.task_done()
                return
            record, chunk = item
            await self._label_chunk(consumer_id, client, record, chunk)

    async def metrics_reporter(self) -> None:
        """Steady-state metrics every 30s — queue depth, in-flight, cps."""
        metrics = self.metrics
        last_processed = 0
        last_t = time.monotonic()
        while not self.stop_event.is_set():
            await asyncio.sleep(30.0)
            now = time.monotonic()
            window_processed = metrics['total_processed'] - last_processed
            window_cps = window_processed / max(now - last_t, 1e-6)
            async with self.in_flight_lock:
                in_flight_count = len(self.in_flight)
            print(
                f'[vlm-worker] metrics: queue={self.queue.qsize()}/{self.queue.maxsize} '
                f'in_flight={in_flight_count} '
                f'window_cps={window_cps:.1f} '
                f'session={metrics["total_processed"]} '
                f'chunks={metrics["total_chunks"]}'
            )
            last_processed = metrics['total_processed']
            last_t = now

    async def run(self) -> None:
        args = self.args
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._on_signal)
        print(
            f'[vlm-worker] streaming: api={args.api} '
            f'vlm_batch={args.vlm_batch_size} concurrency={args.concurrency} '
            f'queue_max={self.queue.maxsize} idle_stop_after={args.idle_stop_after}'
        )
        async with httpx.AsyncClient() as client:
            prod_task = asyncio.create_task(self.producer(), name='producer')
            prod_task.add_done_callback(_crash_on_unhandled_exception)
            cons_tasks = [
                asyncio.create_task(self.consumer(i, client), name=f'consumer-{i}')
                for i in range(args.concurrency)
            ]
            for t in cons_tasks:
                t.add_done_callback(_crash_on_unhandled_exception)
            metrics_task = asyncio.create_task(self.metrics_reporter())
            heartbeat_task = asyncio.create_task(
                heartbeat_loop(
                    'vlm_worker',
                    lambda: {
                        'producer': not prod_task.done(),
                        'consumers': any(not t.done() for t in cons_tasks),
                    },
                    self.stop_event,
                    stall_fn=lambda: self.stall_reason,
                )
            )

            # Wait for either signal-stop or producer-drain.
            await self.stop_event.wait()
            # Wait for producer to finish current iteration.
            await prod_task
            # Drain the queue: wait for consumers to finish what's already
            # queued, then send poison pills.
            await self.queue.join()
            for _ in range(args.concurrency):
                await self.queue.put(None)
            await asyncio.gather(*cons_tasks, return_exceptions=True)
            metrics_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await metrics_task
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        await self.opensearch.close()

        elapsed_total = time.monotonic() - self.started_at
        rate_total = self.metrics['total_processed'] / max(elapsed_total, 1e-6)
        print(
            f'[vlm-worker] done: processed={self.metrics["total_processed"]} '
            f'updated={self.metrics["total_updated"]} chunks={self.metrics["total_chunks"]} '
            f'elapsed={elapsed_total:.1f}s avg_rate={rate_total:.1f} cps'
        )


def _crash_on_unhandled_exception(task: asyncio.Task) -> None:
    """A task dying silently (e.g. an import error inside the
    producer coroutine) previously left the worker reporting
    `session=0 chunks=0` forever with a passing healthcheck. Any
    task that finishes with an exception other than cancellation is
    fatal — log the traceback and exit so the container restart
    policy recovers the worker.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    logging.getLogger('vlm-worker').critical(
        'fatal: task %s exited with an unhandled exception', task.get_name(), exc_info=exc
    )
    os._exit(1)


async def run(args: argparse.Namespace) -> int:
    """Streaming producer/consumer pipeline across every served project.

    Replaces the old burst pattern (fetch -> gather all -> repeat) with
    a continuous flow:

      Producer task: pulls chunks of ``vlm_batch_size`` crop_ids
        from OpenSearch and feeds an asyncio.Queue. Sleeps when the
        queue is full (backpressure) or when OpenSearch returns
        nothing. Tracks an ``in_flight`` set so it doesn't re-fetch
        crops the consumers haven't finished updating yet (the OS
        ``must_not`` query only excludes them after the API writes
        their terminal class_source). Each cycle splits the fetch evenly
        over the active, unpaused projects, rotating the first one.

      Consumer tasks (N = ``--concurrency``): each pulls a chunk
        from the queue, calls that chunk's project's scoped
        ``label_batch``, and removes
        the crop_ids from ``in_flight``. They never wait on each other
        or on the producer — vLLM stays continuously fed.

    Result: vLLM's ``Running:`` count stays steady at ~max-num-seqs
    instead of bursting between 0 and 60.
    """
    await _StreamWorker(args).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    from src.config.retired_env import reject_retired_env

    reject_retired_env()
    args = parse_args(argv)
    if args.project:
        # Fails fast on an unknown/unbindable slug.
        bind_script_project(args.project, opensearch_url=args.opensearch)
    start_worker_metrics_server(os.environ.get('OP_VLM_WORKER_METRICS_PORT', '4610'))
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())

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
    docker compose exec yolo-api python scripts/curation/vlm_worker.py --until-empty
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from scripts.curation._project_worker_utils import (
    curation_api_prefix,
    rotated,
    scoped_url,
    unpaused_projects,
)
from src.config.project_context import bind_project
from src.services.curation.embedding_state import embedded_clause
from src.services.curation.worker_liveness import heartbeat_loop
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import (
    add_project_argument,
    bind_script_project,
    script_project_registry,
)


DEFAULT_API = os.environ.get('OP_API', 'http://localhost:4603')
DEFAULT_OS = os.environ.get('OPENSEARCH_URL', 'http://localhost:4607')
# How long a released-then-not-yet-refreshed crop id stays in the
# released_at guard. Mirrors scripts/curation/worker/runner.py's
# _RELEASED_AT_TTL_S.
_RELEASED_AT_TTL_S = 300.0

# Default thresholds match ``pipeline_auto_label``'s skip logic so the
# worker and the on-demand pipeline make the same decisions.
# Skip VLM classify when the classifier already labeled the crop with at
# least this confidence. Raised 0.70 -> 0.80 to align with
# pipeline.classifier_confidence_skip_vlm and the detection worker's combined
# path's own low-confidence threshold. The 0.70-0.80 band was sending high-confidence classifier crops
# to the VLM and surfacing them in the vlm_low_conf review tab as
# "classifier 95.9 %, VLM medium" — noise the human review queue doesn't need.
DEFAULT_CLASSIFIER_CONF_SKIP = 0.80


_classifier_sources_empty_warned = False


def _warn_classifier_sources_empty_once() -> None:
    """Log once (not every poll) that the 'classifier already confident'
    must_not guard is inactive because classifier_class_sources() is
    empty in this environment."""
    global _classifier_sources_empty_warned  # noqa: PLW0603 - warn-once flag
    if not _classifier_sources_empty_warned:
        _classifier_sources_empty_warned = True
        print(
            '[vlm-worker] classifier_class_sources() is empty; the '
            "'classifier already confident' skip guard is inactive -- "
            'every crop is a VLM candidate regardless of classifier confidence.'
        )


def _build_pending_query(classifier_skip_conf: float, exclude_ids: list[str] | None = None) -> dict:
    """Crops that need the VLM right now.

    Mirrors the ``must_not`` clauses in pipeline_auto_label so the same
    crops the on-demand pipeline would process are picked up by the worker.

    ``exclude_ids`` pushes the producer's in-flight set into the
    query server-side (``must_not: {ids: ...}``) instead of over-fetching
    ``batch_size + len(in_flight)`` docs and filtering in-flight ids out
    in Python.
    """
    # Lazy: keeps the module import light; src.config is all this pulls in.
    from src.services.curation.ingest_class_sources import (
        OPEN_VOCAB_TARGET_CLASS_SOURCE,
        classifier_class_sources,
    )
    from src.services.curation.vlm_class_attempt import (
        VLM_CLASS_ATTEMPTED_AT_FIELD,
        recent_empty_answer_clause,
    )

    must_not: list[dict] = [
        {'term': {'class_validated': True}},
        # Asked recently and the answer had no class: not again until the
        # retry window passes (the class fields are untouched, so nothing
        # else keeps the item out of this query).
        recent_empty_answer_clause(),
    ]
    classifier_sources = sorted(classifier_class_sources())
    if classifier_sources:
        # classifier already confident
        must_not.append(
            {
                'bool': {
                    'filter': [
                        {'terms': {'class_source': classifier_sources}},
                        {'range': {'confidence': {'gte': classifier_skip_conf}}},
                    ],
                },
            },
        )
    else:
        # An empty terms clause matches nothing (correct as a
        # must_not exclusion) but is dead weight in the query shape --
        # only emit it when there's something to exclude, and log once
        # so operators know this guard rail is inactive in this env.
        _warn_classifier_sources_empty_once()
    # One `terms` clause instead of 5 separate `term` clauses on the
    # same field — same match semantics, one less clause for OS to eval.
    must_not.append(
        {
            'terms': {
                'class_source': [
                    # prototype + ensemble_proto_rescue +
                    # ensemble_consensus class_source values are gone.
                    # Surviving auto-validation path is
                    # 'classifier_vlm_agreement' (A-PR2 ensemble writer;
                    # this query excludes already-labeled rows).
                    # VLM already labeled successfully:
                    'vlm',
                    'classifier_vlm_agreement',
                    'cluster_majority_agreement',
                    # VLM already failed once — won't help to retry:
                    'vlm_unmatched',
                    'vlm_new_class_pending',
                ],
            },
        },
    )
    # A user-named open-vocabulary class is never relabeled; the VLM is asked
    # once so its answer can ride along as a suggestion.
    must_not.append(
        {
            'bool': {
                'filter': [
                    {'terms': {'class_source': [OPEN_VOCAB_TARGET_CLASS_SOURCE]}},
                    {'exists': {'field': VLM_CLASS_ATTEMPTED_AT_FIELD}},
                ],
            },
        },
    )
    if exclude_ids:
        must_not.append({'ids': {'values': exclude_ids}})
    return {
        'bool': {
            'filter': [embedded_clause()],
            'must_not': must_not,
        },
    }


def _filter_fresh_ids[K](
    ids: list[K],
    *,
    in_flight: set[K],
    released_at: dict[K, float],
    fetch_started: float,
) -> list[K]:
    """Ids a producer may safely dispatch: not currently in flight, and not
    released at or after ``fetch_started``.

    A fetch that started before (or at the same moment as) a consumer's
    release may still observe pre-write state, since the write uses
    ``refresh=False``. Only an id released strictly *before* this fetch
    began is guaranteed fresh.
    """
    return [
        i for i in ids if i not in in_flight and released_at.get(i, float('-inf')) < fetch_started
    ]


async def fetch_pending_ids(
    opensearch: Any,
    *,
    batch_size: int,
    classifier_skip_conf: float,
    exclude_ids: list[str] | None = None,
) -> list[str]:
    """Pull up to ``batch_size`` crop IDs of the BOUND project that need the VLM.

    ``_source: False`` returns ids only. (Not ``stored_fields: '_none_'``:
    OpenSearch drops the ``_id`` metadata field with it too.)
    ``track_total_hits: False`` skips the exact-count pass this producer
    never reads. ``exclude_ids`` (the caller's in-flight set) is pushed
    into the query itself instead of being filtered out in Python after
    over-fetching ``batch_size + len(in_flight)`` docs.
    """
    from src.config.curation import items_index

    body = {
        'size': batch_size,
        '_source': False,
        'track_total_hits': False,
        'query': _build_pending_query(classifier_skip_conf, exclude_ids=exclude_ids),
        # Oldest pending first — fairness across crops added across the
        # run; also avoids head-of-line starvation when new crops keep
        # arriving from ingest. crop_id tiebreaker keeps paging stable
        # for same-timestamp crops.
        'sort': [{'created_at': {'order': 'asc', 'unmapped_type': 'date'}}, {'crop_id': 'asc'}],
    }
    resp = await opensearch.search(index=items_index(), body=body)
    return [h['_id'] for h in resp.get('hits', {}).get('hits', [])]


def _error_code(response: httpx.Response) -> str | None:
    try:
        detail = response.json().get('detail')
    except ValueError:
        return None
    return detail.get('error') if isinstance(detail, dict) else None


async def label_batch(
    client: httpx.AsyncClient,
    *,
    api: str,
    api_prefix: str,
    slug: str,
    crop_ids: list[str],
) -> dict:
    """Call the scoped /curation/projects/{slug}/vlm/label_batch for one chunk.

    Returns ``{'no_classes': True}`` when the project has no classes yet,
    so the caller backs off instead of logging a failure per chunk."""
    r = await client.post(
        scoped_url(api, api_prefix, slug, '/vlm/label_batch'),
        json={'crop_ids': crop_ids},
        timeout=300.0,
    )
    if r.status_code == 409 and _error_code(r) == 'no_classes':
        return {'no_classes': True}
    r.raise_for_status()
    return r.json()


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
    api_prefix = curation_api_prefix()
    registry = script_project_registry(args.opensearch)
    opensearch = make_script_opensearch([args.opensearch], timeout=30)
    rotation = 0

    stop_event = asyncio.Event()

    def _on_signal(*_: object) -> None:
        if not stop_event.is_set():
            print('[vlm-worker] stop requested — finishing in-flight chunks...')
            stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    started_at = time.monotonic()
    metrics = {
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
    in_flight: set[tuple[str, str]] = set()
    in_flight_lock = asyncio.Lock()
    # label_batch writes with refresh=False, so a producer fetch
    # that starts right after a consumer discards a crop from in_flight
    # can still see the pre-write state and re-dispatch it (duplicate GPU
    # work). Ported from the region/SAM worker's runner.py pattern: hold
    # each released id here for one refresh interval past
    # _RELEASED_AT_TTL_S, keyed by release time, and require a fetch to
    # have started after that release to treat the id as fresh again.
    released_at: dict[tuple[str, str], float] = {}

    # Queue depth: small buffer between producer and consumers. Just
    # big enough to absorb one OS fetch latency. Larger doesn't help
    # — vLLM's max-num-seqs caps real throughput downstream.
    queue: asyncio.Queue[tuple[Any, list[str]] | None] = asyncio.Queue(maxsize=args.concurrency * 2)

    print(
        f'[vlm-worker] streaming: api={args.api} '
        f'vlm_batch={args.vlm_batch_size} concurrency={args.concurrency} '
        f'queue_max={queue.maxsize} idle_stop_after={args.idle_stop_after}'
    )

    async def _fetch_project(record: Any, n: int) -> list[str]:
        """This project's fresh pending ids, read under its own binding."""
        fetch_started = time.monotonic()
        async with in_flight_lock:
            exclude_ids = [cid for slug, cid in in_flight if slug == record.slug]
        with bind_project(record):
            ids = await fetch_pending_ids(
                opensearch,
                batch_size=n,
                classifier_skip_conf=args.classifier_conf_skip,
                exclude_ids=exclude_ids,
            )
        # Drop ids still in flight, and ids released since (or shortly
        # before) this fetch started -- the write used refresh=False, so
        # a fetch that began around the release may still see stale state.
        async with in_flight_lock:
            fresh = _filter_fresh_ids(
                [(record.slug, cid) for cid in ids],
                in_flight=in_flight,
                released_at=released_at,
                fetch_started=fetch_started,
            )
            horizon = fetch_started - _RELEASED_AT_TTL_S
            for key in [k for k, ts in released_at.items() if ts < horizon]:
                del released_at[key]
        return [cid for _, cid in fresh]

    async def producer() -> None:
        """Continuously fetch eligible crops, project by project, and chunk
        them into the queue."""
        nonlocal rotation
        while not stop_event.is_set():
            # Pause if the GPU arbiter says so (training claimed the GPU).
            if args.pause_sentinel and Path(args.pause_sentinel).exists():
                await asyncio.sleep(args.pause_poll_interval)
                continue
            # Backpressure: don't outpace the consumers.
            if queue.full():
                await asyncio.sleep(0.05)
                continue
            projects = rotated(await unpaused_projects(registry, args.project), rotation)
            rotation += 1
            # In-flight ids are excluded server-side (must_not ids), so a
            # fetch only needs to refill the queue.
            fetch_n = min(args.vlm_batch_size * args.concurrency * 2, 9000)  # OS hits cap
            share = max(1, fetch_n // max(1, len(projects)))
            fetched_any = False
            for record in projects:
                try:
                    fresh = await _fetch_project(record, share)
                except Exception as exc:
                    print(f'[vlm-worker] project={record.slug} fetch error: {exc}')
                    continue
                if not fresh:
                    continue
                fetched_any = True
                # Chunk + queue. Mark in-flight before queueing so a fast
                # consumer can't race a slow OS write.
                for i in range(0, len(fresh), args.vlm_batch_size):
                    chunk = fresh[i : i + args.vlm_batch_size]
                    async with in_flight_lock:
                        in_flight.update((record.slug, cid) for cid in chunk)
                    await queue.put((record, chunk))

            if not fetched_any:
                metrics['consecutive_empty_polls'] += 1
                if not args.continuous and (
                    args.until_empty or metrics['consecutive_empty_polls'] >= args.idle_stop_after
                ):
                    # Drain: signal consumers to stop after the queue empties.
                    print(
                        f'[vlm-worker] producer idle '
                        f'({metrics["consecutive_empty_polls"]} empty polls) — draining'
                    )
                    stop_event.set()
                    return
                await asyncio.sleep(args.poll_interval)
                continue
            metrics['consecutive_empty_polls'] = 0

    async def consumer(consumer_id: int, client: httpx.AsyncClient) -> None:
        """Pull a chunk from the queue, call its project's label_batch, repeat."""
        while True:
            item = await queue.get()
            if item is None:  # poison pill = drain complete
                queue.task_done()
                return
            record, chunk = item
            t0 = time.monotonic()
            try:
                result = await label_batch(
                    client, api=args.api, api_prefix=api_prefix, slug=record.slug, crop_ids=chunk
                )
                if result.get('no_classes'):
                    if not metrics.get('no_classes_logged'):
                        print('[vlm-worker] project has no classes yet; idling until some exist')
                        metrics['no_classes_logged'] = 1
                    await asyncio.sleep(args.poll_interval)
                    continue
                metrics['no_classes_logged'] = 0
                metrics['total_processed'] += int(result.get('predicted', 0))
                metrics['total_updated'] += int(result.get('updated', 0))
                metrics['total_chunks'] += 1
                rate = len(chunk) / max(time.monotonic() - t0, 1e-6)
                # Throttle the per-chunk log to 1-in-10 to keep logs readable.
                if metrics['total_chunks'] % 10 == 0:
                    elapsed_total = time.monotonic() - started_at
                    rate_avg = metrics['total_processed'] / max(elapsed_total, 1e-6)
                    print(
                        f'[vlm-worker] chunk {metrics["total_chunks"]} '
                        f'(consumer {consumer_id}): {len(chunk)} crops in '
                        f'{time.monotonic() - t0:.1f}s ({rate:.1f} cps)  '
                        f'session={metrics["total_processed"]} '
                        f'avg={rate_avg:.1f} cps'
                    )
            except httpx.HTTPError as exc:
                print(f'[vlm-worker] consumer {consumer_id} HTTP error: {exc}')
            except Exception as exc:
                print(f'[vlm-worker] consumer {consumer_id} error: {exc}')
            finally:
                released = time.monotonic()
                async with in_flight_lock:
                    for cid in chunk:
                        in_flight.discard((record.slug, cid))
                        released_at[(record.slug, cid)] = released
                queue.task_done()

    async def metrics_reporter() -> None:
        """Steady-state metrics every 30s — queue depth, in-flight, cps."""
        last_processed = 0
        last_t = time.monotonic()
        while not stop_event.is_set():
            await asyncio.sleep(30.0)
            now = time.monotonic()
            window_processed = metrics['total_processed'] - last_processed
            window_cps = window_processed / max(now - last_t, 1e-6)
            async with in_flight_lock:
                in_flight_count = len(in_flight)
            print(
                f'[vlm-worker] metrics: queue={queue.qsize()}/{queue.maxsize} '
                f'in_flight={in_flight_count} '
                f'window_cps={window_cps:.1f} '
                f'session={metrics["total_processed"]} '
                f'chunks={metrics["total_chunks"]}'
            )
            last_processed = metrics['total_processed']
            last_t = now

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

    async with httpx.AsyncClient() as client:
        prod_task = asyncio.create_task(producer(), name='producer')
        prod_task.add_done_callback(_crash_on_unhandled_exception)
        cons_tasks = [
            asyncio.create_task(consumer(i, client), name=f'consumer-{i}')
            for i in range(args.concurrency)
        ]
        for t in cons_tasks:
            t.add_done_callback(_crash_on_unhandled_exception)
        metrics_task = asyncio.create_task(metrics_reporter())
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(
                'vlm_worker',
                lambda: {
                    'producer': not prod_task.done(),
                    'consumers': any(not t.done() for t in cons_tasks),
                },
                stop_event,
            )
        )

        # Wait for either signal-stop or producer-drain.
        await stop_event.wait()
        # Wait for producer to finish current iteration.
        await prod_task
        # Drain the queue: wait for consumers to finish what's already
        # queued, then send poison pills.
        await queue.join()
        for _ in range(args.concurrency):
            await queue.put(None)
        await asyncio.gather(*cons_tasks, return_exceptions=True)
        metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await metrics_task
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
    await opensearch.close()

    elapsed_total = time.monotonic() - started_at
    rate_total = metrics['total_processed'] / max(elapsed_total, 1e-6)
    print(
        f'[vlm-worker] done: processed={metrics["total_processed"]} '
        f'updated={metrics["total_updated"]} chunks={metrics["total_chunks"]} '
        f'elapsed={elapsed_total:.1f}s avg_rate={rate_total:.1f} cps'
    )

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Async polling worker that drains uncertain crops to the VLM.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--api', default=DEFAULT_API, help='triton-api base URL')
    p.add_argument('--opensearch', default=DEFAULT_OS, help='OpenSearch base URL')
    p.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Crops fetched per poll; must be >= vlm_batch_size * concurrency.',
    )
    p.add_argument(
        '--vlm-batch-size',
        type=int,
        default=32,
        help='Crops per label_batch call.',
    )
    p.add_argument(
        '--concurrency',
        type=int,
        default=8,
        help='Concurrent label_batch calls in flight (single-project mode only).',
    )
    p.add_argument(
        '--poll-interval',
        type=float,
        default=5.0,
        help='Seconds to wait when the queue is empty before polling again.',
    )
    p.add_argument(
        '--idle-stop-after',
        type=int,
        default=6,
        help='Consecutive empty polls before the worker exits.',
    )
    p.add_argument(
        '--until-empty',
        action='store_true',
        help='One drain pass: exit on the first empty poll regardless of idle-stop-after.',
    )
    p.add_argument(
        '--continuous',
        action='store_true',
        help='Run forever -- never exit on idle (single-project mode only).',
    )
    p.add_argument(
        '--classifier-conf-skip',
        '--classifier-conf-skip',
        dest='classifier_conf_skip',
        type=float,
        default=DEFAULT_CLASSIFIER_CONF_SKIP,
        help='Skip the VLM for classifier-labeled crops at or above this confidence.',
    )
    # GPU arbiter sentinel (single-project mode): pauses while the
    # trainer holds a single-GPU run. See src/services/training/gpu_arbiter.py.
    p.add_argument(
        '--pause-sentinel',
        default=os.environ.get(
            'OP_PAUSE_SENTINEL',
            str(
                Path(os.environ.get('OP_STATE_DIR', '/var/lib/openprocessor'))
                / 'vlm_worker'
                / 'pause.sentinel'
            ),
        ),
        help='File whose presence pauses the worker (GPU arbiter writes it).',
    )
    p.add_argument(
        '--pause-poll-interval',
        type=float,
        default=10.0,
        help='Seconds to sleep between sentinel checks while paused.',
    )
    add_project_argument(p)
    # Default: every active project; --project restricts to one.
    p.set_defaults(project=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.config.retired_env import reject_retired_env

    reject_retired_env()
    args = parse_args(argv)
    if args.project:
        # Fails fast on an unknown/unbindable slug.
        bind_script_project(args.project, opensearch_url=args.opensearch)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())

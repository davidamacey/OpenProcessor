#!/usr/bin/env python3
"""Async Gemma labeling worker — overlaps Gemma (GPU 2) with ingest (GPU 0).

Background loop that polls OpenSearch for crops that need Gemma's vision
verdict and dispatches them to /curation/vlm/label_batch in parallel with
ongoing image ingestion. This unsticks the pipeline at HDD scale where
the previous "ingest everything → then Gemma" sequence wasted 50%+ of
elapsed time waiting for one GPU while the other was idle.

Selection criteria match ``pipeline_auto_label``'s skip logic exactly so
behavior is consistent: process crops where v6 was uncertain, prototype
assignment was borderline, YOLO11 found a vehicle v6 didn't recognize,
or HDBSCAN clustered the crop as residual. Skip crops where v6 was
confident (Gemma adds no signal there) and crops Gemma has already
processed (asking again won't help).

Operations
----------
- Idempotent: every successful Gemma response writes one of the class_source
  values that the worker's must_not query excludes (``gemma``,
  ``classifier_vlm_agreement``, ``vlm_unmatched``, ``vlm_new_class_pending``),
  so the crop drops out of the next poll's query.
- Auto-exits when ``--idle-stop-after`` consecutive empty polls happen,
  so it can be chained after an ingest run without a sentinel signal.
- Stoppable with SIGINT / SIGTERM — finishes the in-flight batch then
  exits cleanly.

Usage
-----
    .venv/bin/python scripts/curation/vlm_worker.py
    .venv/bin/python scripts/curation/vlm_worker.py --batch-size 32 --concurrency 4
    .venv/bin/python scripts/curation/vlm_worker.py --until-empty   # one drain pass

Or via the Make target:
    make curation-vlm-worker
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
import time
from pathlib import Path

import httpx


DEFAULT_API = os.environ.get('OP_API', 'http://localhost:4603')
# Same env + default as CurationConfig.api_prefix, so the worker follows the API's mount.
API_PREFIX = os.environ.get('OP_API_PREFIX', '/curation').rstrip('/')
DEFAULT_OS = os.environ.get('OPENSEARCH_URL', 'http://localhost:4607')
# This script polls OpenSearch directly (bypassing yolo-api), so it needs
# the same override the src/ modules read via
# src.clients.curation_opensearch / src.config.CurationConfig.items_index.
# Kept as a bare os.environ.get (no src import) so this lightweight
# httpx-only worker doesn't pull in the full src.clients import chain.
ITEMS_INDEX = os.environ.get('OP_ITEMS_INDEX_OVERRIDE') or 'op_items'
# Mirrors src.config.curation.ITEM_EMBEDDING_FIELD for the same no-src-import
# reason; tests/curation/test_item_embedding_field.py pins the two together.
ITEM_EMBEDDING_FIELD = 'pe_embedding'

# F-11: how long a released-then-not-yet-refreshed crop id stays in the
# released_at guard. Mirrors scripts/curation/worker/runner.py's
# _RELEASED_AT_TTL_S.
_RELEASED_AT_TTL_S = 300.0

# Default thresholds match ``pipeline_auto_label``'s skip logic so the
# worker and the on-demand pipeline make the same decisions.
# Skip Gemma classify when v6 model already labeled the crop with at
# least this confidence. Raised 0.70 -> 0.80 to align with
# pipeline.classifier_confidence_skip_vlm and the detection worker's combined
# path's own low-confidence threshold. The 0.70-0.80 band was sending high-v6 crops
# to Gemma and surfacing them in the vlm_low_conf review tab as
# "v6 95.9 %, gemma medium" — noise the human review queue doesn't need.
# See docs/design/plate_detection_strategy.md Wave 1 chained tuning.
DEFAULT_V6_CONF_SKIP = 0.80


_classifier_sources_empty_warned = False


def _warn_classifier_sources_empty_once() -> None:
    """Log once (not every poll) that the 'classifier already confident'
    must_not guard is inactive because classifier_class_sources() is
    empty in this environment (F-11)."""
    global _classifier_sources_empty_warned  # noqa: PLW0603 - warn-once flag
    if not _classifier_sources_empty_warned:
        _classifier_sources_empty_warned = True
        print(
            '[vlm-worker] classifier_class_sources() is empty; the '
            "'classifier already confident' skip guard is inactive -- "
            'every crop is a VLM candidate regardless of classifier confidence.'
        )


def _build_pending_query(v6_skip_conf: float, exclude_ids: list[str] | None = None) -> dict:
    """Crops that need the VLM right now.

    Mirrors the ``must_not`` clauses in pipeline_auto_label so the same
    crops the on-demand pipeline would process are picked up by the worker.

    F-20: ``exclude_ids`` pushes the producer's in-flight set into the
    query server-side (``must_not: {ids: ...}``) instead of over-fetching
    ``batch_size + len(in_flight)`` docs and filtering in-flight ids out
    in Python.
    """
    # Lazy: keeps the module import light; src.config is all this pulls in.
    from src.services.curation.ingest_class_sources import classifier_class_sources

    must_not: list[dict] = [{'term': {'class_validated': True}}]
    classifier_sources = sorted(classifier_class_sources())
    if classifier_sources:
        # classifier already confident
        must_not.append(
            {
                'bool': {
                    'filter': [
                        {'terms': {'class_source': classifier_sources}},
                        {'range': {'confidence': {'gte': v6_skip_conf}}},
                    ],
                },
            },
        )
    else:
        # F-11: an empty terms clause matches nothing (correct as a
        # must_not exclusion) but is dead weight in the query shape --
        # only emit it when there's something to exclude, and log once
        # so operators know this guard rail is inactive in this env.
        _warn_classifier_sources_empty_once()
    # F-20: one `terms` clause instead of 5 separate `term` clauses on the
    # same field — same match semantics, one less clause for OS to eval.
    must_not.append(
        {
            'terms': {
                'class_source': [
                    # Plan §1.5: prototype + ensemble_proto_rescue +
                    # ensemble_consensus class_source values are gone.
                    # Surviving auto-validation path is
                    # 'classifier_vlm_agreement' (A-PR2 ensemble writer;
                    # this query excludes already-labeled rows).
                    # Gemma already labeled successfully:
                    'vlm',
                    'classifier_vlm_agreement',
                    'cluster_majority_agreement',
                    # Gemma already failed once — won't help to retry:
                    'vlm_unmatched',
                    'vlm_new_class_pending',
                ],
            },
        },
    )
    if exclude_ids:
        must_not.append({'ids': {'values': exclude_ids}})
    return {
        'bool': {
            'filter': [{'exists': {'field': ITEM_EMBEDDING_FIELD}}],
            'must_not': must_not,
        },
    }


def _filter_fresh_ids(
    ids: list[str],
    *,
    in_flight: set[str],
    released_at: dict[str, float],
    fetch_started: float,
) -> list[str]:
    """Ids a producer may safely dispatch: not currently in flight, and not
    released at or after ``fetch_started`` (F-11).

    A fetch that started before (or at the same moment as) a consumer's
    release may still observe pre-write state, since the write uses
    ``refresh=False``. Only an id released strictly *before* this fetch
    began is guaranteed fresh.
    """
    return [
        i for i in ids if i not in in_flight and released_at.get(i, float('-inf')) < fetch_started
    ]


async def fetch_pending_ids(
    client: httpx.AsyncClient,
    *,
    opensearch_url: str,
    batch_size: int,
    v6_skip_conf: float,
    exclude_ids: list[str] | None = None,
) -> list[str]:
    """Pull up to ``batch_size`` crop IDs that need Gemma.

    F-20: ``stored_fields: '_none_'`` skips loading the stored document
    entirely (only ``_id``, always free metadata, is returned) — cheaper
    than the previous ``_source: False`` for the same "ids only" result.
    ``track_total_hits: False`` skips the exact-count pass this producer
    never reads. ``exclude_ids`` (the caller's in-flight set) is pushed
    into the query itself instead of being filtered out in Python after
    over-fetching ``batch_size + len(in_flight)`` docs.
    """
    body = {
        'size': batch_size,
        'stored_fields': '_none_',
        'track_total_hits': False,
        'query': _build_pending_query(v6_skip_conf, exclude_ids=exclude_ids),
        # Oldest pending first — fairness across crops added across the
        # run; also avoids head-of-line starvation when new crops keep
        # arriving from ingest. crop_id tiebreaker keeps paging stable
        # for same-timestamp crops.
        'sort': [{'created_at': {'order': 'asc', 'unmapped_type': 'date'}}, {'crop_id': 'asc'}],
    }
    r = await client.post(
        f'{opensearch_url}/{ITEMS_INDEX}/_search',
        json=body,
        timeout=30.0,
    )
    r.raise_for_status()
    return [h['_id'] for h in r.json().get('hits', {}).get('hits', [])]


async def label_batch(
    client: httpx.AsyncClient,
    *,
    api: str,
    crop_ids: list[str],
) -> dict:
    """Call /curation/vlm/label_batch for one chunk."""
    r = await client.post(
        f'{api}{API_PREFIX}/vlm/label_batch',
        json={'crop_ids': crop_ids},
        timeout=300.0,
    )
    r.raise_for_status()
    return r.json()


async def run(args: argparse.Namespace) -> int:
    """Streaming producer/consumer pipeline.

    Replaces the old burst pattern (fetch -> gather all -> repeat) with
    a continuous flow:

      Producer task: pulls chunks of ``vlm_batch_size`` crop_ids
        from OpenSearch and feeds an asyncio.Queue. Sleeps when the
        queue is full (backpressure) or when OpenSearch returns
        nothing. Tracks an ``in_flight`` set so it doesn't re-fetch
        crops the consumers haven't finished updating yet (the OS
        ``must_not`` query only excludes them after the API writes
        their terminal class_source).

      Consumer tasks (N = ``--concurrency``): each pulls a chunk
        from the queue, calls ``/curation/vlm/label_batch``, and removes
        the crop_ids from ``in_flight``. They never wait on each other
        or on the producer — vLLM stays continuously fed.

    Result: vLLM's ``Running:`` count stays steady at ~max-num-seqs
    instead of bursting between 0 and 60.
    """
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
    in_flight: set[str] = set()
    in_flight_lock = asyncio.Lock()
    # F-11: label_batch writes with refresh=False, so a producer fetch
    # that starts right after a consumer discards a crop from in_flight
    # can still see the pre-write state and re-dispatch it (duplicate GPU
    # work). Ported from the region/SAM worker's runner.py pattern: hold
    # each released id here for one refresh interval past
    # _RELEASED_AT_TTL_S, keyed by release time, and require a fetch to
    # have started after that release to treat the id as fresh again.
    released_at: dict[str, float] = {}

    # Queue depth: small buffer between producer and consumers. Just
    # big enough to absorb one OS fetch latency. Larger doesn't help
    # — vLLM's max-num-seqs caps real throughput downstream.
    queue: asyncio.Queue[list[str] | None] = asyncio.Queue(maxsize=args.concurrency * 2)

    print(
        f'[vlm-worker] streaming: api={args.api} '
        f'vlm_batch={args.vlm_batch_size} concurrency={args.concurrency} '
        f'queue_max={queue.maxsize} idle_stop_after={args.idle_stop_after}'
    )

    async def producer(client: httpx.AsyncClient) -> None:
        """Continuously fetch eligible crops and chunk them into the queue.

        Same in-flight-skipping bug fix as sam-worker: fetch beyond the
        in-flight window so we don't keep re-fetching the same oldest
        ids that consumers are still processing.
        """
        while not stop_event.is_set():
            # Pause if the GPU arbiter says so (training claimed the GPU).
            if args.pause_sentinel and Path(args.pause_sentinel).exists():
                await asyncio.sleep(args.pause_poll_interval)
                continue
            # Backpressure: don't outpace the consumers.
            if queue.full():
                await asyncio.sleep(0.05)
                continue
            try:
                # F-20: in-flight ids are excluded server-side (must_not
                # ids) now, so the fetch only needs to refill the queue —
                # no more "+ in_flight_count" over-fetch-then-filter. The
                # 1000+ in-flight ids at concurrency=24 still ride along
                # as a must_not clause, which OS evaluates as a cheap
                # docvalue lookup rather than as extra hits to transfer
                # and discard.
                fetch_n = min(args.vlm_batch_size * args.concurrency * 2, 9000)  # OS hits cap
                fetch_started = time.monotonic()
                async with in_flight_lock:
                    exclude_ids = list(in_flight)
                ids = await fetch_pending_ids(
                    client,
                    opensearch_url=args.opensearch,
                    batch_size=fetch_n,
                    v6_skip_conf=args.v6_conf_skip,
                    exclude_ids=exclude_ids,
                )
            except httpx.HTTPError as exc:
                print(f'[vlm-worker] producer fetch error: {exc}')
                await asyncio.sleep(args.poll_interval)
                continue

            # Filter out ids the consumers are still processing, and ids
            # released since (or shortly before) this fetch started -- the
            # write used refresh=False, so a fetch that began around the
            # same time as the release may still see stale state (F-11).
            async with in_flight_lock:
                fresh = _filter_fresh_ids(
                    ids, in_flight=in_flight, released_at=released_at, fetch_started=fetch_started
                )
                horizon = fetch_started - _RELEASED_AT_TTL_S
                for cid in [c for c, ts in released_at.items() if ts < horizon]:
                    del released_at[cid]

            if not fresh:
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

            # Chunk + queue. Mark in-flight before queueing so a fast
            # consumer can't race a slow OS write.
            for i in range(0, len(fresh), args.vlm_batch_size):
                chunk = fresh[i : i + args.vlm_batch_size]
                async with in_flight_lock:
                    in_flight.update(chunk)
                await queue.put(chunk)

    async def consumer(consumer_id: int, client: httpx.AsyncClient) -> None:
        """Pull a chunk from the queue, call /curation/vlm/label_batch, repeat."""
        while True:
            chunk = await queue.get()
            if chunk is None:  # poison pill = drain complete
                queue.task_done()
                return
            t0 = time.monotonic()
            try:
                result = await label_batch(client, api=args.api, crop_ids=chunk)
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
                        in_flight.discard(cid)
                        released_at[cid] = released
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

    async with httpx.AsyncClient() as client:
        prod_task = asyncio.create_task(producer(client))
        cons_tasks = [asyncio.create_task(consumer(i, client)) for i in range(args.concurrency)]
        metrics_task = asyncio.create_task(metrics_reporter())

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
        description='Async polling worker that drains uncertain crops to Gemma.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--api', default=DEFAULT_API, help='triton-api base URL')
    p.add_argument('--opensearch', default=DEFAULT_OS, help='OpenSearch base URL')
    p.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help=(
            'Crops fetched per poll. CRITICAL: must be >= vlm_batch_size * '
            'concurrency, otherwise chunking yields too few chunks and the '
            'concurrency semaphore is wasted (worker runs serial). 256 '
            'with default vlm-batch=32 + concurrency=8 yields 8 '
            'concurrent chunks — saturates vLLM at ~42 cps.'
        ),
    )
    p.add_argument(
        '--vlm-batch-size',
        type=int,
        default=32,
        help=(
            'Crops per /curation/vlm/label_batch call. 32 = 8 upstream Gemma calls '
            'per HTTP roundtrip; balances per-call overhead against head-of-line '
            'blocking on slow chunks.'
        ),
    )
    p.add_argument(
        '--concurrency',
        type=int,
        default=8,
        help=(
            'Concurrent /curation/vlm/label_batch calls in flight. With '
            'vlm-batch-size=32 + concurrency=8 we keep ~256 crop slots in '
            'flight, which after 4-img chunking lands around 64 in-flight '
            "upstream — close to vLLM's ~42 cps peak (max-num-seqs=64 + "
            '--enable-prefix-caching). See test_results/gemma_bench/.'
        ),
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
        help=(
            'Number of consecutive empty polls before the worker exits. '
            'At default poll-interval=5s, 6 polls ≈ 30s of idleness.'
        ),
    )
    p.add_argument(
        '--until-empty',
        action='store_true',
        help='One drain pass: exit on the first empty poll regardless of idle-stop-after.',
    )
    p.add_argument(
        '--continuous',
        action='store_true',
        help=(
            'Run forever — never exit on idle. Use this when running as a '
            'long-lived background service (Docker compose, systemd, tmux). '
            'Polls every poll-interval seconds until SIGINT/SIGTERM.'
        ),
    )
    p.add_argument(
        '--classifier-conf-skip',
        '--v6-conf-skip',
        dest='v6_conf_skip',
        type=float,
        default=DEFAULT_V6_CONF_SKIP,
        help='Skip the VLM for classifier-labeled crops at or above this confidence.',
    )
    # GPU arbiter sentinel — design §14.5. The trainer touches this file
    # before a single-GPU run starts; the worker pauses while it exists
    # so we don't fight the trainer for CPU/RAM. (For dual-GPU runs the
    # arbiter stops the whole gemma container instead, so this path
    # never runs.) See src/services/training/gpu_arbiter.py.
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())

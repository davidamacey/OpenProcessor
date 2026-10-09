"""Producer, writer and metrics reporter of the streaming pipeline."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import time
from typing import TYPE_CHECKING, Any

from scripts.curation.worker.bulk_writer import _bulk_update
from scripts.curation.worker.fairness import (
    fetch_pending_multi_project,
    is_project_paused,
    write_liveness,
)
from scripts.curation.worker.region_embed_stage import embed_written_regions
from scripts.curation.worker.state import _ItemTask, _wait_for_sentinel_clear
from src.core.logging import get_logger
from src.services.curation.crop_bytes import cache_stats as crop_cache_stats
from src.services.curation.ops_metrics import OP_DETECTION_WORKER_ITEMS_TOTAL


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from scripts.curation.worker.pipeline import PipelineContext


logger = get_logger('curation_worker')


# How long the producer remembers a released crop. Only has to outlive the
# slowest single pending search.
_RELEASED_AT_TTL_S = 300.0


async def producer(
    ctx: PipelineContext, sync_project_runtime: Callable[[Any], Awaitable[None]]
) -> None:
    """Fetch eligible crops and queue task DESCRIPTORS only.

    In-flight crops are now excluded server-side (``must_not
    ids``), so the fetch no longer needs to over-fetch
    ``batch_size + in_flight_count`` and then filter in Python — the
    old bug this over-fetch fixed (batch_size=96, in_flight=132, the
    96 oldest all in-flight, fresh=0) can't recur when the query
    itself already excludes in-flight ids.
    """
    args = ctx.args
    stop_event = ctx.stop_event
    opensearch = ctx.opensearch
    sentinel = ctx.sentinel
    project_registry = ctx.project_registry
    fairness_scheduler = ctx.fairness_scheduler
    in_q = ctx.in_q
    vlm_visible_q = ctx.vlm_visible_q
    sam_q = ctx.sam_q
    combined_q = ctx.combined_q
    out_q = ctx.out_q
    in_flight = ctx.in_flight
    in_flight_owner = ctx.in_flight_owner
    in_flight_lock = ctx.in_flight_lock
    released_at = ctx.released_at
    metrics = ctx.metrics
    # A project's share of the whole pipeline (every inter-stage
    # queue), so one project with a slow leg cannot fill it; with a
    # single project the cap is the pipeline itself, as before.
    pipeline_capacity = sum(q.maxsize for q in (in_q, vlm_visible_q, sam_q, combined_q, out_q))
    while not stop_event.is_set():
        await _wait_for_sentinel_clear(sentinel, sleep_s=args.sentinel_sleep)
        if stop_event.is_set():
            return
        # Backpressure: don't overfill the queue.
        if in_q.full():
            await asyncio.sleep(0.05)
            continue
        # OpenSearch caps a single search hits at 10000 by default;
        # stay safely under that.
        fetch_n = min(args.batch_size, 9000)
        fetch_started = time.monotonic()
        async with in_flight_lock:
            exclude_ids = list(in_flight)
            for cid in [c for c in in_flight_owner if c not in in_flight]:
                del in_flight_owner[cid]
            inflight_counts = collections.Counter(in_flight_owner.values())
        fairness_scheduler.set_in_flight(dict(inflight_counts))
        await project_registry.ensure_fresh()

        # W2 sec 4.5 (B1): one hot-reload check per active project,
        # each bound in turn -- never a single process-wide runtime.
        # A brand-new project (first cycle it is seen) builds its
        # first runtime here too; a project whose sync fails this
        # cycle just keeps its last-known runtime (or none) and is
        # retried next cycle.
        for _record in project_registry.active_projects():
            await sync_project_runtime(_record)

        try:
            tasks = await fetch_pending_multi_project(
                opensearch,
                registry=project_registry,
                scheduler=fairness_scheduler,
                fetch_n=fetch_n,
                queue_max=pipeline_capacity,
                exclude_ids=exclude_ids,
            )
        except Exception as exc:
            logger.warning('producer_fetch_error', error=str(exc))
            await asyncio.sleep(args.poll_interval)
            continue

        # Filter out in-flight tasks and hits that may predate a write
        # released while this search was running; keep only fresh ones.
        async with in_flight_lock:
            fresh = [
                t
                for t in tasks
                if t.crop_id not in in_flight
                and released_at.get(t.crop_id, float('-inf')) < fetch_started
            ]
            horizon = fetch_started - _RELEASED_AT_TTL_S
            for cid in [c for c, ts in released_at.items() if ts < horizon]:
                del released_at[cid]

        if not fresh:
            metrics['consecutive_empty_polls'] += 1
            if not args.continuous:
                logger.info(
                    'producer_idle_exiting',
                    empty_polls=metrics['consecutive_empty_polls'],
                )
                stop_event.set()
                return
            await asyncio.sleep(args.poll_interval)
            continue
        metrics['consecutive_empty_polls'] = 0

        # Mark in-flight then queue task descriptors. Cap how many
        # we queue per fetch — we don't want to flood in_q in a
        # single iteration since each consumer can only consume
        # one at a time. Backpressure (in_q.full check at top of
        # loop) handles the rest.
        async with in_flight_lock:
            for t in fresh[: args.batch_size]:
                in_flight.add(t.crop_id)
                in_flight_owner[t.crop_id] = t.project.slug
            inflight_counts = collections.Counter(in_flight_owner.values())
        for t in fresh[: args.batch_size]:
            await in_q.put(t)

        # Liveness (§5.1): one runtime record per active project.
        for p in project_registry.active_projects():
            with contextlib.suppress(OSError):
                write_liveness(
                    p,
                    inflight=inflight_counts.get(p.slug, 0),
                    applied=True,
                    paused=is_project_paused(p),
                )


async def writer(ctx: PipelineContext) -> None:
    """Drain out_q, bulk-update OpenSearch every WRITE_FLUSH_INTERVAL or full.

    Decouples GPU consumer throughput from OS write latency. Without
    this, every consumer that finishes blocks on its next iteration
    until OS bulk_update returns; with this, GPU work continues
    while the writer batches OS writes.
    """
    args = ctx.args
    opensearch = ctx.opensearch
    started_at = ctx.started_at
    region_embed_pe = ctx.region_embed_pe
    out_q = ctx.out_q
    in_flight = ctx.in_flight
    in_flight_lock = ctx.in_flight_lock
    released_at = ctx.released_at
    metrics = ctx.metrics
    visible_no_verdict = ctx.visible_no_verdict
    combined_no_verdict = ctx.combined_no_verdict
    WRITE_FLUSH_SIZE = max(args.batch_size // 2, 32)
    WRITE_FLUSH_INTERVAL = 1.0  # seconds
    pending: list[_ItemTask] = []
    last_flush = time.monotonic()

    async def _flush(reason: str) -> None:
        nonlocal last_flush
        if not pending:
            return
        # B3: snapshot + clear `pending` up front, and call
        # `out_q.task_done()` once per item HERE -- only once its
        # write has actually completed (or definitively failed) --
        # not at `get()` time. `quiesce_and_swap`'s drain
        # (`out_q.join()`) is the guarantee that every item queued
        # before a swap is durably written (and stamped with the
        # OLD runtime's refs, since `_bulk_update` reads the store
        # under the item's own project binding, which does not move
        # until the swap that is BLOCKED on this same join()).
        # Calling `task_done()` at `get()` time let the join()
        # return while flushes for old-runtime items were still
        # sitting unflushed in `pending`, so they were written --
        # and stamped -- after the swap already happened.
        flushing = list(pending)
        pending.clear()
        t0 = time.monotonic()
        batch_request_ids = [t.request_id for t in flushing]
        if region_embed_pe is not None:
            await embed_written_regions(flushing, region_embed_pe)
        try:
            n_written, n_skipped = await _bulk_update(opensearch, flushing)
        except Exception as exc:
            logger.warning(
                'writer_bulk_update_failed',
                n=len(flushing),
                request_ids=batch_request_ids,
                error=str(exc),
            )
            # Drop these from in_flight so they get re-fetched by
            # the producer on the next pass.
            async with in_flight_lock:
                for t in flushing:
                    in_flight.discard(t.crop_id)
            OP_DETECTION_WORKER_ITEMS_TOTAL.labels(outcome='failed').inc(len(flushing))
            for _ in flushing:
                out_q.task_done()
            last_flush = time.monotonic()
            return
        OP_DETECTION_WORKER_ITEMS_TOTAL.labels(outcome='ok').inc(n_written)
        if n_skipped:
            OP_DETECTION_WORKER_ITEMS_TOTAL.labels(outcome='skipped').inc(n_skipped)
        metrics['total_processed'] += len(flushing)
        metrics['total_written'] += n_written
        elapsed = time.monotonic() - t0
        rate = metrics['total_processed'] / max(time.monotonic() - started_at, 1e-6)
        logger.info(
            'region_worker_flush',
            reason=reason,
            flushed=len(flushing),
            written=n_written,
            skipped=n_skipped,
            flush_s=round(elapsed, 2),
            session_processed=metrics['total_processed'],
            session_avg_cps=round(rate, 2),
            request_ids=batch_request_ids,
        )
        # Now safe to remove from in_flight: the bulk ran with
        # refresh='wait_for', so searches started from here on see it.
        released = time.monotonic()
        async with in_flight_lock:
            for t in flushing:
                in_flight.discard(t.crop_id)
                released_at[t.crop_id] = released
                visible_no_verdict.clear(t.crop_id)
                combined_no_verdict.clear(t.crop_id)
        for _ in flushing:
            out_q.task_done()
        last_flush = time.monotonic()

    try:
        while True:
            # Wait up to flush_interval seconds for the next item;
            # if it times out and we have anything pending, flush.
            try:
                timeout = max(0.05, WRITE_FLUSH_INTERVAL - (time.monotonic() - last_flush))
                t = await asyncio.wait_for(out_q.get(), timeout=timeout)
            except TimeoutError:
                await _flush('interval')
                continue
            if t is None:
                out_q.task_done()
                await _flush('shutdown')
                return
            pending.append(t)
            if len(pending) >= WRITE_FLUSH_SIZE:
                await _flush('size')
    except asyncio.CancelledError:
        await _flush('cancelled')
        raise


async def metrics_reporter(ctx: PipelineContext) -> None:
    """Emit a single-line steady-state metrics dump every 30s.

    Surfaces queue depths + combined-call outcomes so operators can
    see whether the secondary segmenter is starved (in_q low), the
    combined VLM call is the wall (combined_q full), or writes are
    the wall (out_q full). ``combined_bbox_wrong`` flags how often
    the VLM sees a region elsewhere in the crop — if it climbs, a
    text-hint retry path may be worth adding back.
    """
    stop_event = ctx.stop_event
    in_q = ctx.in_q
    vlm_visible_q = ctx.vlm_visible_q
    sam_q = ctx.sam_q
    combined_q = ctx.combined_q
    out_q = ctx.out_q
    in_flight = ctx.in_flight
    in_flight_lock = ctx.in_flight_lock
    metrics = ctx.metrics
    visible_no_verdict = ctx.visible_no_verdict
    combined_no_verdict = ctx.combined_no_verdict
    crop_gate = ctx.crop_gate
    last_processed = 0
    last_t = time.monotonic()
    while not stop_event.is_set():
        await asyncio.sleep(30.0)
        now = time.monotonic()
        window_processed = metrics['total_processed'] - last_processed
        window_cps = window_processed / max(now - last_t, 1e-6)
        cache_hits, cache_misses = crop_cache_stats()
        cache_total = cache_hits + cache_misses
        hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0
        async with in_flight_lock:
            in_flight_count = len(in_flight)
        vis_total = metrics['vlm_visible_kept'] + metrics['vlm_visible_skipped']
        vis_skip_rate = metrics['vlm_visible_skipped'] / vis_total if vis_total > 0 else 0.0
        logger.info(
            'region_worker_metrics',
            in_q_depth=in_q.qsize(),
            in_q_max=in_q.maxsize,
            vlm_visible_q_depth=vlm_visible_q.qsize(),
            vlm_visible_q_max=vlm_visible_q.maxsize,
            sam_q_depth=sam_q.qsize(),
            sam_q_max=sam_q.maxsize,
            combined_q_depth=combined_q.qsize(),
            combined_q_max=combined_q.maxsize,
            out_q_depth=out_q.qsize(),
            out_q_max=out_q.maxsize,
            in_flight=in_flight_count,
            window_cps=round(window_cps, 2),
            session_processed=metrics['total_processed'],
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate=round(hit_rate, 3),
            vlm_visible_kept=metrics['vlm_visible_kept'],
            vlm_visible_skipped=metrics['vlm_visible_skipped'],
            vlm_visible_skip_rate=round(vis_skip_rate, 3),
            region_gate_skipped=crop_gate.skipped_total,
            visible_no_verdict=metrics['visible_no_verdict'],
            visible_no_verdict_cap_hits=metrics['visible_no_verdict_cap_hits'],
            combined_bbox_wrong=metrics['combined_bbox_wrong'],
            combined_no_region_visible=metrics['combined_no_region_visible'],
            combined_parse_failure=metrics['combined_parse_failure'],
            combined_no_bbox_verdict=metrics['combined_no_bbox_verdict'],
            combined_no_verdict_cap_hits=metrics['combined_no_verdict_cap_hits'],
            no_verdict_tracked=len(visible_no_verdict) + len(combined_no_verdict),
        )
        last_processed = metrics['total_processed']
        last_t = now

"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/sam_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

# ruff: noqa: E402
import asyncio
import contextlib
import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.metrics import (
    LEGACY_STAGE_A_GEMMA_VISIBLE_DURATION_SECONDS,
    LEGACY_STAGE_A_SAM_DURATION_SECONDS,
    LEGACY_STAGE_B_GEMMA_VERIFY_DURATION_SECONDS,
    LEGACY_STAGE_LPR_DURATION_SECONDS,
)
from src.services.detection.cascade_detect import (
    REFERENCE_LICENSE_PLATE_PROFILE,
    PaddleOcrTextRecognizer,
    RegionDetector,
    crop_norm_to_source_norm,
    is_plausible_region_bbox,
)
from src.services.labeling.vlm_labeler import CombinedCrop, RegionCrop


logger = get_logger('curation_worker')


from scripts.curation.worker import state
from scripts.curation.worker.bulk_writer import _bulk_update
from scripts.curation.worker.cascade import (
    Sam3AllHostsDown,
    _fetch_pending,
    _resegment_from_text_hint,
    _source_to_crop,
)
from scripts.curation.worker.state import (
    _PENDING_DETECTION_ALIASES,
    _PENDING_VERIFICATION_ALIASES,
    _TERMINAL_STATUSES,
    _crop_jpeg_for_task,
    _is_secondary_shape,
    _ItemTask,
    _wait_for_sentinel_clear,
)
from scripts.curation.worker.verify import (
    _SKIP_VLM_VERIFY_SECONDARY_SCORE,
    _auto_confirm_or_pending,
    _bbox_shape_is_plausible,
    _combined_class_update,
    _combined_write_doc,
    _region_write_doc,
)


if TYPE_CHECKING:
    import argparse

    from aiohttp import web


async def _start_metrics_http_server(*, port: int) -> web.AppRunner:
    """Stand up a minimal aiohttp /metrics endpoint inside the worker.

    The worker process otherwise has no HTTP surface, so Prometheus
    cannot scrape its in-process registry. This helper exposes the
    process-default :mod:`prometheus_client` registry — which is the
    same registry the stage histograms register on at import time —
    at ``GET /metrics``.

    Returned :class:`aiohttp.web.AppRunner` is the cleanup handle the
    caller stops on shutdown.
    """
    from aiohttp import web as _web
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    async def _metrics_handler(_request: _web.Request) -> _web.Response:
        return _web.Response(body=generate_latest(), content_type=CONTENT_TYPE_LATEST.split(';')[0])

    app = _web.Application()
    app.router.add_get('/metrics', _metrics_handler)
    runner = _web.AppRunner(app, access_log=None)
    await runner.setup()
    site = _web.TCPSite(runner, host='0.0.0.0', port=port)
    await site.start()
    logger.info('sam_worker_metrics_server_started', port=port)
    return runner


# High-conf primary-classifier cohort skips classification in the
# combined call (the caller already has a trusted class). Same
# threshold as the legacy cascade's combined-cohort gate
# (combined.py: _V6_LOW_CONF_THRESHOLD).
_V6_HIGH_CONF_CLASS_SOURCES = frozenset({'v6_model', 'cluster_v6_majority_agreement'})
_V6_HIGH_CONF_THRESHOLD = 0.80


def _should_classify(t: _ItemTask, *, registry_loaded: bool) -> bool:
    """Decide whether to ask the VLM for the item class on this crop.

    Module-level (not a ``run()`` closure) so it's directly unit-testable
    — see ``tests/curation/test_write_guards.py``.

    Returns False (skip classification) when the registry didn't load,
    the crop was already confirmed by a human (P0-2 human-label
    guard — a human verdict must never be silently reclassified), the
    crop is a frozen test_holdout crop (P0-3 — class-field guard only;
    region fields stay unconditional, see ``_build_pending_query``), OR
    the crop already carries a high-confidence primary / cluster-primary
    class label. Otherwise returns True and the combined prompt asks
    the VLM to fill ``class_id``.
    """
    if not registry_loaded:
        return False
    if t.class_validated or t.class_source.startswith('human'):
        return False
    if t.test_holdout:
        return False
    return not (
        t.class_source in _V6_HIGH_CONF_CLASS_SOURCES
        and t.class_confidence >= _V6_HIGH_CONF_THRESHOLD
    )


async def run(args: argparse.Namespace) -> int:
    """Streaming producer/consumer pipeline.

    Replaces the burst pattern (fetch -> gather all -> bulk write -> repeat)
    with a continuous flow:

      Producer task: pulls _ItemTask batches from OpenSearch, builds the
        crop JPEG in a thread (RAM cache or HDD fallback), and feeds a
        bounded asyncio.Queue. Tracks an ``in_flight`` set so it
        doesn't re-fetch crops the consumers haven't finished updating
        in OS yet.

      Writer task: drains completed tasks via asyncio.Queue and bulk-
        updates OpenSearch on a small interval (or when a soft-batch
        threshold hits). Decoupled from consumers so OS writes don't
        block GPU work.

      Consumer tasks (N = ``--concurrency``): pull from input queue,
        run the segmenter + VLM + OCR chain in ``_process_crop``, push
        the result to the writer queue.

    Result: the secondary-segmenter GPU and the shared VLM stay
    continuously fed; the OS poll + bulk-write phases overlap with GPU
    work instead of stopping it.
    """
    stop_event = asyncio.Event()

    def _on_signal(*_: object) -> None:
        if not stop_event.is_set():
            logger.info('stop_requested')
            stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    # Look up heavy-IO constructors through the legacy shim module so
    # tests that monkeypatch `scripts.curation.sam_worker_main.AsyncTritonPool`
    # (and friends) still intercept calls made from this split-out runner.
    from scripts.curation import sam_worker_main as _wkr

    pool = _wkr.AsyncTritonPool(url=args.triton, pool_size=args.pool_size, max_concurrent=64)
    await pool.initialize()
    lpr = RegionDetector(pool)
    # text-hinted re-pass: when the primary detector + secondary
    # segmenter both globally miss but the VLM confirmed the crop has a
    # region of interest, run the OCR pipeline (det + rec) on the whole
    # crop, pick the region-shaped text region, then re-prompt the
    # segmenter with a tight sub-crop around it. The segmenter produces
    # the final geometry — the OCR-detection bbox is never trusted as a
    # region bbox source (it's too loose; produced visibly-oversized
    # regions).
    ocr_recognizer = PaddleOcrTextRecognizer(pool)
    # D5: the segmenter leg is optional. An empty ``--sam3-url``/``SAM3_URL``
    # constructs a disabled Sam3Client — segment_plate() then always
    # returns None (the same "no candidate" result callers already
    # handle) without attempting any HTTP call. A deployment with no
    # segmentation service of its own leaves this unset.
    sam3 = _wkr.Sam3Client(args.sam3_url)
    gemma = _wkr.VlmLabeler(base_url=args.gemma_url) if args.gemma_url else _wkr.VlmLabeler()
    # B-PR5: populate class_names so ``label_combined`` callers (the
    # primary-detector-missed cohort gate in cascade._process_crop) can
    # classify in the same VLM round-trip as region verify + OCR.
    # Best-effort: if the registry can't be loaded the cohort gate
    # falls back to legacy two-call paths (label_combined with empty
    # class_names just answers the region side).
    name_to_id: dict[str, int] = {}
    try:
        from src.clients.curation_opensearch import ClassRegistry

        _reg = ClassRegistry().load()
        # gemma.class_names is the list passed into the VLM prompt;
        # reply.class_id is the *index* into this list, NOT the
        # registry id. name_to_id maps the resolved name back to the
        # registry's authoritative class_id so writes carry the
        # correct value. Without this remap, a reply of class_id=0
        # lands the registry's first non-deprecated class label on a
        # doc with class_id=0 (deprecated) — historical drift.
        gemma.class_names = [c.class_name for c in _reg.classes if not c.deprecated]
        name_to_id = {c.class_name: int(c.class_id) for c in _reg.classes if not c.deprecated}
        gemma.name_to_id = name_to_id
    except Exception as _exc:  # nosec B110 — best-effort, registry optional
        logger.warning('class_registry_load_failed', error=str(_exc))
    opensearch = _wkr.AsyncOpenSearch(hosts=[args.opensearch])

    started_at = time.monotonic()
    sentinel = Path(args.pause_sentinel)

    in_flight: set[str] = set()
    in_flight_lock = asyncio.Lock()

    # Pipeline (hybrid: ≤ 2 VLM round-trips per crop, with the cheap
    # visibility pre-filter shielding the slow segmenter GPU and the
    # heavy combined call from non-region crops):
    #
    #   in_q             -> Stage A.primary (pending_verify + primary-
    #                       detector Triton call)
    #                       splits into:
    #                         - has candidate         -> combined_q
    #                         - primary miss / secondary-shape -> gemma_visible_q
    #
    #   gemma_visible_q  -> Stage A.gemma_visible (batched yes/no
    #                       VISIBLE_CHUNK per call, fail-OPEN on parse error)
    #                       splits into:
    #                         - visible=True          -> sam_q
    #                         - visible=False         -> out_q (no_region_visible)
    #
    #   sam_q            -> Stage A.secondary + text-hint OCR re-pass
    #                       splits into:
    #                         - high-conf skip        -> out_q (detected)
    #                         - candidate found        -> combined_q
    #                         - all detectors miss     -> out_q (no_region_box)
    #
    #   combined_q       -> Stage B (batched combined VLM call,
    #                       COMBINED_CHUNK per call)
    #                       writes detected / verify_rejected / no_region_visible
    #                       to out_q
    #
    #   out_q            -> writer (batched OS bulk_update)
    #
    # Visible filter rationale: the secondary segmenter is the
    # bottleneck. A yes/no VLM call is cheap batched but skips the
    # segmenter + combined call for every "no region visible" crop.
    # Even a modest skip rate recovers more segmenter + VLM budget than
    # the pre-filter consumes. Primary-hit crops bypass the filter
    # entirely (we already know there's a region); only primary-miss +
    # secondary-shape crops walk it.
    #
    # Inter-stage queues are sized large so the slow side never starves
    # the fast side. RAM cost is ~30 KB JPEG per task, so even 2k
    # queued = ~60 MB.
    in_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=args.concurrency * 2)
    gemma_visible_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=2000)
    sam_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=args.concurrency * 2)
    combined_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=2000)
    out_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=args.concurrency * 8)
    # Number of Stage B (combined VLM) consumers. Each consumer drains
    # combined_q in chunks of COMBINED_CHUNK (6 per upstream call) so
    # the effective VLM concurrency = consumers x chunk = 16 x 6 = 96
    # in-flight. The removed visibility stage no longer competes for
    # VLM slots, so we can spend the full VLM budget on the combined
    # call (which subsumes both old calls).
    gemma_concurrency = int(os.environ.get('SAM_WORKER_GEMMA_CONCURRENCY', '16'))
    # Visibility pre-filter: cheap yes/no, packed VISIBLE_CHUNK per call.
    # Default 8 consumers gives 8 x 6 = 48 in-flight calls at the
    # upstream VLM, well under the combined-call budget of 16 x 6 = 96.
    # The visible filter is fast (~2s/call) so it doesn't need as many
    # consumers as the heavier combined call.
    gemma_visible_concurrency = int(os.environ.get('SAM_WORKER_GEMMA_VISIBLE_CONCURRENCY', '8'))
    # Aligned with the shared VLM's --limit-mm-per-prompt {"image":6}.
    # Per-call work scales worse than linearly past 6 on the reference
    # deployment's GPU for this prompt+image mix.
    COMBINED_CHUNK = 6
    # Same per-call image budget for the visibility filter (the
    # upstream --limit-mm-per-prompt cap is shared across both prompts).
    VISIBLE_CHUNK = 6
    # Wait at most this long for a chunk to fill before firing it
    # under-full. Keeps tail latency bounded when the queue empties out
    # near end-of-run while still benefiting from batching during
    # steady-state.
    GEMMA_CHUNK_DRAIN_TIMEOUT = 0.10

    metrics = {
        'total_processed': 0,
        'total_written': 0,
        'consecutive_empty_polls': 0,
        # gemma_visible_skipped: primary-miss / secondary-shape crops
        # that the visibility pre-filter short-circuited to
        # no_region_visible (no segmenter call, no combined call). The
        # point of this stage; bigger is better.
        'gemma_visible_skipped': 0,
        # gemma_visible_kept: crops that passed the filter and went on
        # to the secondary-segmenter stage. Together with skipped, lets
        # us compute the filter's skip rate at a glance.
        'gemma_visible_kept': 0,
        # combined_bbox_wrong: the VLM said the region IS visible but
        # the proposed bbox was wrong. We write verify_rejected and do
        # NOT re-loop the segmenter (avoids re-introducing a 2nd VLM
        # call). If this counter climbs high, future work could add a
        # text-hint retry path.
        'combined_bbox_wrong': 0,
        # combined_parse_failure: per-crop entry missing or unparseable
        # in the batched response. Crop is dropped from in_flight so
        # the next producer poll re-fetches it (no terminal status
        # stamped).
        'combined_parse_failure': 0,
        # combined_no_plate_visible: the VLM confirmed no region is
        # visible at all. Terminal write.
        'combined_no_plate_visible': 0,
    }

    logger.info(
        'sam_worker_start_streaming',
        opensearch=args.opensearch,
        triton=args.triton,
        sam3_url=args.sam3_url,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        continuous=args.continuous,
        max_iterations=args.max_iterations,
    )

    async def producer() -> None:
        """Fetch eligible crops and queue task DESCRIPTORS only.

        Critical: fetch SIZE must exceed the in_flight set so we don't
        keep re-fetching the same oldest crops that are already being
        processed by Stage A or Stage B. Bug observed in production:
        with batch_size=96 and in_flight=132 (mostly stuck in Stage B
        waiting for the VLM), fetch returned the 96 oldest crops which
        were ALL in_flight, so fresh=0 every poll → Stage A starved.

        Fix: fetch batch_size + in_flight_count so we always get at
        least batch_size beyond the currently-processing window.
        """
        while not stop_event.is_set():
            await _wait_for_sentinel_clear(sentinel, sleep_s=args.sentinel_sleep)
            if stop_event.is_set():
                return
            # Backpressure: don't overfill the queue.
            if in_q.full():
                await asyncio.sleep(0.05)
                continue
            # Size the fetch to skip past everything currently in flight.
            # The OS query returns oldest-first, so we need to fetch
            # past the ages of in_flight crops to get to fresh ones.
            async with in_flight_lock:
                in_flight_count = len(in_flight)
            fetch_n = args.batch_size + in_flight_count
            # OpenSearch caps a single search hits at 10000 by default.
            # If in_flight ever blows past that, we'd need search_after
            # or scroll. For now cap at 9000 to stay safely under.
            fetch_n = min(fetch_n, 9000)
            try:
                tasks = await _fetch_pending(opensearch, batch_size=fetch_n)
            except Exception as exc:
                logger.warning('producer_fetch_error', error=str(exc))
                await asyncio.sleep(args.poll_interval)
                continue

            # Filter out in-flight tasks; keep only fresh ones.
            async with in_flight_lock:
                fresh = [t for t in tasks if t.crop_id not in in_flight]

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
            for t in fresh[: args.batch_size]:
                await in_q.put(t)

    async def stage_a_consumer(consumer_id: int) -> None:
        """Stage A.primary: load JPEG + primary/pending_verify routing only.

        Never calls the VLM or the secondary segmenter directly. Routes:
          - pending_verify (existing primary-detector candidate) -> combined_q
          - pending + non-secondary-shape with primary hit       -> combined_q
          - pending + secondary-shape OR primary miss             -> gemma_visible_q
            (let the VLM decide if a region is even visible before
            we spend the slow segmenter + combined call on it;
            primary-hit crops already proved a region is present
            and bypass the filter)

        Decoupling the primary detector (fast Triton) from the
        secondary segmenter (slow GPU) keeps the primary-detector
        Triton instances saturated during long segmenter / VLM stalls.
        """
        while True:
            t = await in_q.get()
            if t is None:
                in_q.task_done()
                return
            # Bind request_id so every structlog event in this iteration
            # carries it (Phase 4a). Cleared in finally so the next task
            # on this consumer task doesn't inherit the previous id.
            structlog.contextvars.bind_contextvars(request_id=t.request_id)
            try:
                # Load JPEG (parallel HDD reads across all consumers).
                if t.crop_jpeg is None:
                    t.crop_jpeg = await asyncio.to_thread(
                        _crop_jpeg_for_task,
                        t.crop_id,
                        t.image_path,
                        t.vehicle_bbox_norm,
                    )
                if t.crop_jpeg is None:
                    F = get_region_fields()
                    t.update_doc = {F.status: RegionStatus.NO_REGION_BOX}
                    await out_q.put(t)
                    in_q.task_done()
                    continue
                if t.plate_status in _TERMINAL_STATUSES:
                    in_q.task_done()
                    continue

                is_secondary = _is_secondary_shape(t)

                # Path 1: pending_verify — already has a primary-detector
                # candidate; straight to combined VLM call (no
                # detection needed).
                if (
                    t.plate_status in _PENDING_VERIFICATION_ALIASES
                    and t.lpr_plate_in_source is not None
                ):
                    t.candidate_source = 'lpr_existing'
                    t.candidate_in_crop = _source_to_crop(
                        t.lpr_plate_in_source, t.vehicle_bbox_norm
                    )
                    t.candidate_in_source = t.lpr_plate_in_source
                    t.candidate_score = t.lpr_score
                    await combined_q.put(t)
                    in_q.task_done()
                    continue

                # Path 2: pending + non-secondary-shape — try the
                # primary detector first (fast Triton call).
                if t.plate_status in _PENDING_DETECTION_ALIASES and not is_secondary:
                    _lpr_t0 = time.monotonic()
                    try:
                        lpr_results = await lpr.detect_batch([t.crop_jpeg])
                    except Exception:
                        LEGACY_STAGE_LPR_DURATION_SECONDS.labels(outcome='error').observe(
                            time.monotonic() - _lpr_t0
                        )
                        raise
                    cand = lpr_results[0] if lpr_results else None
                    LEGACY_STAGE_LPR_DURATION_SECONDS.labels(
                        outcome='hit' if cand is not None else 'miss'
                    ).observe(time.monotonic() - _lpr_t0)
                    if cand is not None:
                        t.candidate_source = 'lpr'
                        t.candidate_in_crop = cand.bbox_norm
                        t.candidate_in_source = crop_norm_to_source_norm(
                            cand.bbox_norm, t.vehicle_bbox_norm
                        )
                        t.candidate_score = cand.score
                        await combined_q.put(t)
                        in_q.task_done()
                        continue

                # Path 3: secondary-shape pending OR non-secondary with
                # no primary hit. Hand off to the visibility
                # pre-filter — VLM yes/no decides whether the slow
                # segmenter + combined path is even worth it. Fails
                # OPEN on parse errors so a flaky VLM never silently
                # drops a real region.
                await gemma_visible_q.put(t)
                in_q.task_done()
            except Exception as exc:
                logger.warning(
                    'stage_a_failed',
                    consumer_id=consumer_id,
                    crop_id=t.crop_id,
                    error=str(exc),
                )
                # Drop in_flight so producer re-fetches; don't write
                # update_doc so OS state stays unchanged.
                async with in_flight_lock:
                    in_flight.discard(t.crop_id)
                in_q.task_done()
            finally:
                structlog.contextvars.unbind_contextvars('request_id')

    async def _drain_chunk(
        q: asyncio.Queue[_ItemTask | None],
        *,
        chunk_size: int,
        drain_timeout: float,
    ) -> tuple[list[_ItemTask], bool]:
        """Pull up to ``chunk_size`` tasks from ``q``.

        Blocks indefinitely on the first task; subsequent tasks are
        non-blocking up to ``drain_timeout`` total. Returns
        ``(tasks, poison_received)``. When ``poison_received`` is True,
        the caller should flush ``tasks`` then exit (the poison pill
        was consumed but is not included in ``tasks``).

        Why batch like this
        -------------------
        The visibility + verify endpoints pack 4 crops per upstream
        VLM call. Pulling one task and immediately firing wastes the
        batching opportunity; waiting forever for chunk_size tasks
        creates terrible tail latency near end-of-run when the queue
        empties out. The bounded drain window balances both.
        """

        tasks: list[_ItemTask] = []
        first = await q.get()
        if first is None:
            q.task_done()
            return tasks, True
        tasks.append(first)
        q.task_done()

        deadline = asyncio.get_running_loop().time() + drain_timeout
        poisoned = False
        while len(tasks) < chunk_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                t = await asyncio.wait_for(q.get(), timeout=remaining)
            except TimeoutError:
                break
            if t is None:
                # Re-queue the poison so siblings can also drain. We
                # only consume one poison pill per consumer, matching
                # the existing N-poison-pills-for-N-consumers shutdown
                # contract upstream.
                q.task_done()
                poisoned = True
                break
            tasks.append(t)
            q.task_done()
        return tasks, poisoned

    async def stage_a_gemma_visible(consumer_id: int) -> None:
        """Stage A.gemma_visible: batched yes/no region-visibility filter.

        Drains ``gemma_visible_q`` in chunks of ``VISIBLE_CHUNK`` and
        asks the VLM "is a region of interest visible at all?" per
        crop. Crops that come back ``False`` short-circuit to
        ``no_region_visible`` without ever touching the secondary
        segmenter or the combined call — that's the entire point of
        this stage. Crops that come back ``True`` (or that the parser
        fails open on) advance to ``sam_q``.

        Fail-OPEN semantics: any RPC error, any per-entry parse
        failure, and any empty response is treated as ``visible=True``
        so we never silently bin a real region when the VLM is flaky.
        The cost is one extra segmenter round-trip on those crops —
        the existing pipeline is the safety net.
        """

        while True:
            chunk, poisoned = await _drain_chunk(
                gemma_visible_q,
                chunk_size=VISIBLE_CHUNK,
                drain_timeout=GEMMA_CHUNK_DRAIN_TIMEOUT,
            )
            if chunk:
                batch_request_ids = [t.request_id for t in chunk]
                plate_crops: list[RegionCrop] = []
                bad_indices: list[int] = []
                for i, t in enumerate(chunk):
                    if t.crop_jpeg is None:
                        bad_indices.append(i)
                        continue
                    plate_crops.append(RegionCrop(crop_id=t.crop_id, jpeg_bytes=t.crop_jpeg))

                verdicts: dict[str, bool] = {}
                if plate_crops:
                    _vis_t0 = time.monotonic()
                    try:
                        verdicts = await gemma.plate_visible_batch(plate_crops)
                        LEGACY_STAGE_A_GEMMA_VISIBLE_DURATION_SECONDS.labels(outcome='ok').observe(
                            time.monotonic() - _vis_t0
                        )
                    except Exception as exc:
                        LEGACY_STAGE_A_GEMMA_VISIBLE_DURATION_SECONDS.labels(outcome='error').observe(
                            time.monotonic() - _vis_t0
                        )
                        logger.warning(
                            'stage_a_gemma_visible_failed',
                            consumer_id=consumer_id,
                            chunk_size=len(plate_crops),
                            request_ids=batch_request_ids,
                            error=str(exc),
                        )
                        # Fail-OPEN: pretend everyone is visible so the
                        # secondary segmenter still gets a crack at them.
                        verdicts = {c.crop_id: True for c in plate_crops}

                F = get_region_fields()
                for i, t in enumerate(chunk):
                    structlog.contextvars.bind_contextvars(request_id=t.request_id)
                    try:
                        if i in bad_indices:
                            t.update_doc = {F.status: RegionStatus.NO_REGION_BOX}
                            await out_q.put(t)
                            continue
                        # Default True (fail-open) when the crop is
                        # missing from the verdicts dict for any reason.
                        is_visible = verdicts.get(t.crop_id, True)
                        if is_visible:
                            metrics['gemma_visible_kept'] += 1
                            t.detection_trace.append('gemma_visible:yes')
                            await sam_q.put(t)
                        else:
                            metrics['gemma_visible_skipped'] += 1
                            t.detection_trace.append('gemma_visible:no')
                            t.update_doc = {
                                F.status: RegionStatus.NO_REGION_VISIBLE,
                                F.detector_chain: list(t.detection_trace),
                            }
                            await out_q.put(t)
                    finally:
                        structlog.contextvars.unbind_contextvars('request_id')
            if poisoned:
                return

    async def stage_a_sam_consumer(consumer_id: int) -> None:
        """Stage A.secondary: secondary-segmenter run + text-hint OCR fallback.

        Runs on every crop the visibility filter said could contain a
        region of interest (primary-miss / secondary-shape crops that
        came through ``stage_a_gemma_visible``). High-confidence
        segmenter hits with a region-shaped bbox short-circuit the VLM
        entirely (zero combined calls — same policy as the pre-collapse
        pipeline; see ``_SKIP_VLM_VERIFY_SECONDARY_SCORE``). Everything
        else routes to Stage B.combined for a single VLM round-trip.
        """

        F = get_region_fields()
        while True:
            t = await sam_q.get()
            if t is None:
                sam_q.task_done()
                return
            structlog.contextvars.bind_contextvars(request_id=t.request_id)
            try:
                if t.crop_jpeg is None:
                    t.update_doc = {F.status: RegionStatus.NO_REGION_BOX}
                    await out_q.put(t)
                    sam_q.task_done()
                    continue

                _sam_t0 = time.monotonic()
                try:
                    sam_candidate = await sam3.segment_plate(t.crop_jpeg)
                except Sam3AllHostsDown as exc:
                    # Infrastructure failure (every secondary-segmenter
                    # host UNHEALTHY). Do NOT mark the crop terminal —
                    # leave plate_status unchanged so it stays in
                    # pending_detection for the next cascade pass once
                    # a host recovers. Drop from in_flight + sleep so
                    # the producer can re-fetch and we don't spin a hot
                    # loop while every host is down.
                    LEGACY_STAGE_A_SAM_DURATION_SECONDS.labels(outcome='error').observe(
                        time.monotonic() - _sam_t0
                    )
                    logger.error(
                        'stage_a_sam_all_hosts_down',
                        consumer_id=consumer_id,
                        crop_id=t.crop_id,
                        error=str(exc),
                    )
                    async with in_flight_lock:
                        in_flight.discard(t.crop_id)
                    sam_q.task_done()
                    await asyncio.sleep(1.0)
                    continue
                except Exception:
                    LEGACY_STAGE_A_SAM_DURATION_SECONDS.labels(outcome='error').observe(
                        time.monotonic() - _sam_t0
                    )
                    raise
                _sam_elapsed = time.monotonic() - _sam_t0
                LEGACY_STAGE_A_SAM_DURATION_SECONDS.labels(
                    outcome='hit' if sam_candidate is not None else 'miss'
                ).observe(_sam_elapsed)
                logger.info(
                    'stage_a_sam_took_ms',
                    crop_id=t.crop_id,
                    ms=round(_sam_elapsed * 1000.0, 2),
                    hit=sam_candidate is not None,
                )
                if sam_candidate is not None:
                    # High-conf-skip: bypass VLM verify on the
                    # strongest hits + valid region shape.
                    if (
                        sam_candidate.score >= _SKIP_VLM_VERIFY_SECONDARY_SCORE
                        and _bbox_shape_is_plausible(sam_candidate.bbox_norm)
                    ):
                        projected = crop_norm_to_source_norm(
                            sam_candidate.bbox_norm, t.vehicle_bbox_norm
                        )
                        t.detection_trace.append(
                            f'{REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name}:hit'
                        )
                        t.detection_trace.append(
                            f'{REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name}:skip_gemma_verify'
                        )
                        t.update_doc = _region_write_doc(
                            plate_in_source=projected,
                            score=sam_candidate.score,
                            detector=REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name,
                            detector_version=REFERENCE_LICENSE_PLATE_PROFILE.segmenter_version,
                            chain=t.detection_trace,
                            plate_verified=False,
                            plate_validated=False,
                            verifier=None,
                            verifier_version=None,
                            extra={F.skip_verify: True},
                        )
                        await out_q.put(t)
                        sam_q.task_done()
                        continue
                    # Else: queue the secondary-segmenter candidate for
                    # combined VLM call.
                    t.candidate_source = 'sam3'
                    t.candidate_in_crop = sam_candidate.bbox_norm
                    t.candidate_in_source = crop_norm_to_source_norm(
                        sam_candidate.bbox_norm, t.vehicle_bbox_norm
                    )
                    t.candidate_score = sam_candidate.score
                    await combined_q.put(t)
                    sam_q.task_done()
                    continue

                # Secondary segmenter missed globally; re-prompt it
                # with a tight sub-crop around an OCR text hint. The
                # OCR-detection bbox is no longer trusted; the
                # segmenter produces the final geometry. OCR text
                # rides along for storage.
                try:
                    ocr_regions = await ocr_recognizer.detect_regions(t.crop_jpeg)
                except Exception as exc:
                    logger.warning('text_hint_ocr_failed', crop_id=t.crop_id, error=str(exc))
                    ocr_regions = []
                ocr_pick = (
                    ocr_recognizer.pick_best_plate_region(ocr_regions) if ocr_regions else None
                )
                if ocr_pick is not None:
                    t.detection_trace.append(
                        f'{REFERENCE_LICENSE_PLATE_PROFILE.ocr_rec_model}:text_hint:hit'
                    )
                    sub_cand, _sub_box = await _resegment_from_text_hint(
                        t.crop_jpeg, ocr_pick.bbox_norm, sam3
                    )
                    if sub_cand is not None:
                        t.candidate_source = 'sam3_text_hint'
                        t.candidate_in_crop = sub_cand.bbox_norm
                        t.candidate_in_source = crop_norm_to_source_norm(
                            sub_cand.bbox_norm, t.vehicle_bbox_norm
                        )
                        t.candidate_score = sub_cand.score
                        t.candidate_text = ocr_pick.text
                        t.candidate_text_confidence = ocr_pick.rec_score
                        await combined_q.put(t)
                        sam_q.task_done()
                        continue
                    t.detection_trace.append(
                        f'{REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name}:text_hint:miss'
                    )
                elif ocr_regions:
                    t.detection_trace.append(
                        f'{REFERENCE_LICENSE_PLATE_PROFILE.ocr_rec_model}:text_hint:no_plate_shape'
                    )
                else:
                    t.detection_trace.append(
                        f'{REFERENCE_LICENSE_PLATE_PROFILE.ocr_rec_model}:text_hint:miss'
                    )

                # Nothing found by any detector → no_region_box.
                t.update_doc = {
                    F.status: RegionStatus.NO_REGION_BOX,
                    F.detector_chain: list(t.detection_trace),
                }
                await out_q.put(t)
                sam_q.task_done()
            except Exception as exc:
                logger.warning(
                    'stage_a_sam_failed',
                    consumer_id=consumer_id,
                    crop_id=t.crop_id,
                    error=str(exc),
                )
                async with in_flight_lock:
                    in_flight.discard(t.crop_id)
                sam_q.task_done()
            finally:
                structlog.contextvars.unbind_contextvars('request_id')

    async def stage_b_combined(consumer_id: int) -> None:
        """Stage B: batched combined VLM call (class + region-verify + OCR).

        Replaces the legacy gemma_visible + gemma_verify two-call
        cascade with ONE VLM round-trip per crop. Each crop's candidate
        bbox (primary or secondary) is drawn as a colored overlay on
        the parent item JPEG before sending so the VLM confirms the
        bbox visually in the same call that classifies the item and
        reads the region text.

        Per-crop branches on the reply:
          - plate_bbox_correct=True, plate_visible=True (+ bbox passes
            sanity gate) -> write 'detected' with full region + class
            fields via :func:`_combined_write_doc`.
          - plate_visible=True but plate_bbox_correct=False (or sanity
            gate fails) -> write 'verify_rejected' + class fields. Do
            NOT re-loop the secondary segmenter (would re-introduce 2
            VLM calls). ``combined_bbox_wrong`` counter tracks this
            cohort.
          - plate_visible=False -> write 'no_region_visible' + class fields.
          - reply missing / parse failure -> drop from in_flight, leave
            plate_status unchanged so the next producer poll re-fetches.
        """

        F = get_region_fields()
        while True:
            chunk, poisoned = await _drain_chunk(
                combined_q,
                chunk_size=COMBINED_CHUNK,
                drain_timeout=GEMMA_CHUNK_DRAIN_TIMEOUT,
            )
            if chunk:
                batch_request_ids = [t.request_id for t in chunk]
                class_names = list(getattr(gemma, 'class_names', None) or [])
                registry_loaded = bool(class_names)

                # Build CombinedCrop payloads (parent item JPEG +
                # candidate bbox for overlay drawing). Per-crop
                # ``classify`` flag tells the VLM whether to fill
                # class_id (low-conf primary / unknown source) or skip
                # it (high-conf primary crops where the caller already
                # trusts the class).
                combined_crops: list[CombinedCrop] = []
                bad_indices: list[int] = []
                for i, t in enumerate(chunk):
                    if t.crop_jpeg is None:
                        bad_indices.append(i)
                        continue
                    combined_crops.append(
                        CombinedCrop(
                            crop_id=t.crop_id,
                            jpeg_bytes=t.crop_jpeg,
                            plate_bbox_norm=t.candidate_in_crop,
                            classify=_should_classify(t, registry_loaded=registry_loaded),
                        )
                    )

                replies_by_id: dict[str, Any] = {}
                if combined_crops:
                    _gemma_t0 = time.monotonic()
                    try:
                        replies_by_id = await gemma.label_combined_batch(
                            combined_crops,
                            class_names=class_names or None,
                        )
                        _gemma_elapsed = time.monotonic() - _gemma_t0
                        LEGACY_STAGE_B_GEMMA_VERIFY_DURATION_SECONDS.labels(outcome='ok').observe(
                            _gemma_elapsed
                        )
                        _gemma_ms = round(_gemma_elapsed * 1000.0, 2)
                        logger.info(
                            'stage_b_combined_took_ms',
                            consumer_id=consumer_id,
                            chunk_size=len(combined_crops),
                            request_ids=batch_request_ids,
                            ms=_gemma_ms,
                            per_crop_ms=round(_gemma_ms / max(1, len(combined_crops)), 2),
                        )
                    except Exception as exc:
                        LEGACY_STAGE_B_GEMMA_VERIFY_DURATION_SECONDS.labels(outcome='error').observe(
                            time.monotonic() - _gemma_t0
                        )
                        logger.warning(
                            'stage_b_combined_failed',
                            consumer_id=consumer_id,
                            chunk_size=len(combined_crops),
                            request_ids=batch_request_ids,
                            error=str(exc),
                        )
                        # Drop everyone in this chunk from in_flight so
                        # the producer re-fetches; don't write any
                        # update_doc so OS state stays unchanged.
                        async with in_flight_lock:
                            for t in chunk:
                                in_flight.discard(t.crop_id)
                        if poisoned:
                            return
                        continue

                for i, t in enumerate(chunk):
                    structlog.contextvars.bind_contextvars(request_id=t.request_id)
                    try:
                        if i in bad_indices:
                            # Defensive — Stage A should always set these.
                            t.update_doc = {F.status: RegionStatus.NO_REGION_BOX}
                            await out_q.put(t)
                            continue
                        reply = replies_by_id.get(t.crop_id)
                        if reply is None:
                            # Per-crop parse failure / missing entry. Leave
                            # in pending — drop from in_flight so the next
                            # producer poll re-fetches.
                            metrics['combined_parse_failure'] += 1
                            logger.warning(
                                'stage_b_combined_parse_failure',
                                crop_id=t.crop_id,
                                request_id=t.request_id,
                            )
                            async with in_flight_lock:
                                in_flight.discard(t.crop_id)
                            continue

                        # Decide per-crop class-field side: only honor
                        # the VLM's class when we asked it to classify.
                        effective_class_names = (
                            class_names
                            if _should_classify(t, registry_loaded=registry_loaded)
                            else None
                        )

                        if (
                            reply.plate_bbox_correct
                            and reply.plate_visible
                            and t.candidate_in_source is not None
                            and t.candidate_in_crop is not None
                        ):
                            # Happy path — the VLM confirmed the bbox is
                            # a real region of interest. Sanity-gate
                            # before committing.
                            gate_ok, gate_reason = is_plausible_region_bbox(
                                t.candidate_in_crop, t.vehicle_bbox_norm
                            )
                            if not gate_ok:
                                t.detection_trace.append(
                                    f'{t.candidate_source or "unknown"}:sanity_reject:{gate_reason}'
                                )
                                t.update_doc = {
                                    F.status: RegionStatus.VERIFY_REJECTED,
                                    F.detector_chain: list(t.detection_trace),
                                    **_combined_class_update(
                                        reply, effective_class_names, name_to_id=name_to_id
                                    ),
                                }
                                await out_q.put(t)
                                continue
                            auto = await _auto_confirm_or_pending(
                                sam_score=t.candidate_score,
                                bbox_in_crop=t.candidate_in_crop,
                                gemma_high_conf=reply.plate_confidence == 'high',
                            )
                            # Map cascade's internal source tag to the
                            # canonical detector name used in provenance.
                            _det = {
                                'sam3': (
                                    REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name,
                                    REFERENCE_LICENSE_PLATE_PROFILE.segmenter_version,
                                ),
                                # OCR-hinted re-pass: bbox came from the
                                # secondary segmenter too, just on a
                                # tighter sub-crop. Provenance records
                                # the segmenter as the detector; the
                                # text-hint trace lives on the detector
                                # chain.
                                'sam3_text_hint': (
                                    REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name,
                                    REFERENCE_LICENSE_PLATE_PROFILE.segmenter_version,
                                ),
                                'lpr': (
                                    REFERENCE_LICENSE_PLATE_PROFILE.detector_model,
                                    REFERENCE_LICENSE_PLATE_PROFILE.detector_version,
                                ),
                                'lpr_existing': (
                                    REFERENCE_LICENSE_PLATE_PROFILE.detector_model,
                                    REFERENCE_LICENSE_PLATE_PROFILE.detector_version,
                                ),
                            }.get(
                                t.candidate_source,
                                (t.candidate_source or 'unknown', '1'),
                            )
                            t.detection_trace.append(f'{_det[0]}:hit')
                            t.detection_trace.append(f'{_det[0]}:combined_verify_ok')
                            t.update_doc = _combined_write_doc(
                                reply=reply,
                                candidate_in_source=t.candidate_in_source,
                                candidate_score=t.candidate_score,
                                detector=_det[0],
                                detector_version=_det[1],
                                chain=t.detection_trace,
                                class_names=effective_class_names,
                                plate_validated=auto or False,
                                name_to_id=name_to_id,
                            )
                            # Preserve the candidate_source marker for
                            # downstream consumers via RegionFields.source.
                            t.update_doc[F.source] = t.candidate_source
                            # Text-hint OCR fallback: the VLM sometimes
                            # reads no text from the bbox when the OCR
                            # recognizer already produced one. Forward
                            # the OCR text so the region is still
                            # searchable.
                            if not reply.plate_text and t.candidate_text:
                                rc = t.candidate_text_confidence or 0.0
                                t.update_doc[F.text] = t.candidate_text
                                t.update_doc[F.text_raw] = t.candidate_text
                                t.update_doc[F.text_source] = (
                                    REFERENCE_LICENSE_PLATE_PROFILE.ocr_rec_model
                                )
                                t.update_doc[F.text_engine_version] = '1'
                                # Map numeric rec_score -> VLM confidence
                                # bin so downstream consumers treat
                                # OCR-sourced confidence consistently.
                                t.update_doc[F.text_confidence] = (
                                    0.92 if rc >= 0.8 else 0.70 if rc >= 0.5 else 0.40
                                )
                        elif reply.plate_visible:
                            # plate_bbox_correct is False/None but the
                            # VLM says a region IS visible. Write
                            # verify_rejected + class fields and do NOT
                            # re-loop the secondary segmenter (would
                            # re-introduce 2 VLM calls per crop). Human
                            # review picks these up.
                            metrics['combined_bbox_wrong'] += 1
                            t.detection_trace.append(
                                f'{t.candidate_source or "unknown"}:'
                                'combined_verify_reject:plate_visible_elsewhere'
                            )
                            t.update_doc = {
                                F.status: RegionStatus.VERIFY_REJECTED,
                                F.detector_chain: list(t.detection_trace),
                                **_combined_class_update(
                                    reply, effective_class_names, name_to_id=name_to_id
                                ),
                            }
                        else:
                            # plate_visible=False — no region in this crop.
                            metrics['combined_no_plate_visible'] += 1
                            t.detection_trace.append(
                                f'{t.candidate_source or "unknown"}:combined_no_plate_visible'
                            )
                            t.update_doc = {
                                F.status: RegionStatus.NO_REGION_VISIBLE,
                                F.detector_chain: list(t.detection_trace),
                                **_combined_class_update(
                                    reply, effective_class_names, name_to_id=name_to_id
                                ),
                            }
                        await out_q.put(t)
                    finally:
                        structlog.contextvars.unbind_contextvars('request_id')
            if poisoned:
                return

    async def writer() -> None:
        """Drain out_q, bulk-update OpenSearch every WRITE_FLUSH_INTERVAL or full.

        Decouples GPU consumer throughput from OS write latency. Without
        this, every consumer that finishes blocks on its next iteration
        until OS bulk_update returns; with this, GPU work continues
        while the writer batches OS writes.
        """
        WRITE_FLUSH_SIZE = max(args.batch_size // 2, 32)
        WRITE_FLUSH_INTERVAL = 1.0  # seconds
        pending: list[_ItemTask] = []
        last_flush = time.monotonic()

        async def _flush(reason: str) -> None:
            nonlocal last_flush
            if not pending:
                return
            t0 = time.monotonic()
            batch_request_ids = [t.request_id for t in pending]
            try:
                n_written, n_skipped = await _bulk_update(opensearch, pending)
            except Exception as exc:
                logger.warning(
                    'writer_bulk_update_failed',
                    n=len(pending),
                    request_ids=batch_request_ids,
                    error=str(exc),
                )
                # Drop these from in_flight so they get re-fetched by
                # the producer on the next pass.
                async with in_flight_lock:
                    for t in pending:
                        in_flight.discard(t.crop_id)
                pending.clear()
                last_flush = time.monotonic()
                return
            metrics['total_processed'] += len(pending)
            metrics['total_written'] += n_written
            elapsed = time.monotonic() - t0
            rate = metrics['total_processed'] / max(time.monotonic() - started_at, 1e-6)
            logger.info(
                'sam_worker_flush',
                reason=reason,
                flushed=len(pending),
                written=n_written,
                skipped=n_skipped,
                flush_s=round(elapsed, 2),
                session_processed=metrics['total_processed'],
                session_avg_cps=round(rate, 2),
                request_ids=batch_request_ids,
            )
            # Now safe to remove from in_flight.
            async with in_flight_lock:
                for t in pending:
                    in_flight.discard(t.crop_id)
            pending.clear()
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
                out_q.task_done()
                if len(pending) >= WRITE_FLUSH_SIZE:
                    await _flush('size')
        except asyncio.CancelledError:
            await _flush('cancelled')
            raise

    async def metrics_reporter() -> None:
        """Emit a single-line steady-state metrics dump every 30s.

        Surfaces queue depths + combined-call outcomes so operators can
        see whether the secondary segmenter is starved (in_q low), the
        combined VLM call is the wall (combined_q full), or writes are
        the wall (out_q full). ``combined_bbox_wrong`` flags how often
        the VLM sees a region elsewhere in the crop — if it climbs, a
        text-hint retry path may be worth adding back.
        """
        last_processed = 0
        last_t = time.monotonic()
        while not stop_event.is_set():
            await asyncio.sleep(30.0)
            now = time.monotonic()
            window_processed = metrics['total_processed'] - last_processed
            window_cps = window_processed / max(now - last_t, 1e-6)
            cache_total = state._cache_hits + state._cache_misses
            hit_rate = state._cache_hits / cache_total if cache_total > 0 else 0.0
            async with in_flight_lock:
                in_flight_count = len(in_flight)
            vis_total = metrics['gemma_visible_kept'] + metrics['gemma_visible_skipped']
            vis_skip_rate = metrics['gemma_visible_skipped'] / vis_total if vis_total > 0 else 0.0
            logger.info(
                'sam_worker_metrics',
                in_q_depth=in_q.qsize(),
                in_q_max=in_q.maxsize,
                gemma_visible_q_depth=gemma_visible_q.qsize(),
                gemma_visible_q_max=gemma_visible_q.maxsize,
                sam_q_depth=sam_q.qsize(),
                sam_q_max=sam_q.maxsize,
                combined_q_depth=combined_q.qsize(),
                combined_q_max=combined_q.maxsize,
                out_q_depth=out_q.qsize(),
                out_q_max=out_q.maxsize,
                in_flight=in_flight_count,
                window_cps=round(window_cps, 2),
                session_processed=metrics['total_processed'],
                cache_hits=state._cache_hits,
                cache_misses=state._cache_misses,
                cache_hit_rate=round(hit_rate, 3),
                gemma_visible_kept=metrics['gemma_visible_kept'],
                gemma_visible_skipped=metrics['gemma_visible_skipped'],
                gemma_visible_skip_rate=round(vis_skip_rate, 3),
                combined_bbox_wrong=metrics['combined_bbox_wrong'],
                combined_no_plate_visible=metrics['combined_no_plate_visible'],
                combined_parse_failure=metrics['combined_parse_failure'],
            )
            last_processed = metrics['total_processed']
            last_t = now

    try:
        prod_task = asyncio.create_task(producer())
        # Stage A.primary pool: primary-detector + pending_verify
        # routing only. Sized to keep the primary-detector Triton
        # instances saturated; secondary-segmenter work has moved to
        # its own pool below.
        stage_a_tasks = [asyncio.create_task(stage_a_consumer(i)) for i in range(args.concurrency)]
        # Stage A.gemma_visible pool: batched yes/no region-visibility
        # filter for primary-miss / secondary-shape crops. Sized
        # smaller than the combined pool because the visibility prompt
        # is cheap.
        stage_a_visible_tasks = [
            asyncio.create_task(stage_a_gemma_visible(i)) for i in range(gemma_visible_concurrency)
        ]
        # Stage A.secondary pool: secondary-segmenter run on every crop
        # the visibility filter said could contain a region. Sized to
        # keep the secondary-segmenter deployment saturated.
        stage_a_sam_tasks = [
            asyncio.create_task(stage_a_sam_consumer(i)) for i in range(args.concurrency)
        ]
        # Stage B pool: batched combined VLM call (COMBINED_CHUNK
        # crops per upstream call). gemma_concurrency x COMBINED_CHUNK
        # in-flight (16 x 6 = 96 by default). Each crop gets at most
        # ONE combined VLM round-trip here (plus at most ONE yes/no
        # call up in the visibility stage = ≤ 2 per crop total).
        stage_b_tasks = [asyncio.create_task(stage_b_combined(i)) for i in range(gemma_concurrency)]
        writer_task = asyncio.create_task(writer())
        metrics_task = asyncio.create_task(metrics_reporter())
        # Phase 4c: stand up an aiohttp /metrics endpoint inside the
        # worker process so Prometheus can scrape the stage-timing
        # histograms (the worker is not an HTTP server otherwise).
        # Port 4609 inside container, mapped 1:1 on the host so the
        # default prometheus scrape config can reach it.
        metrics_server_runner = await _start_metrics_http_server(
            port=int(os.environ.get('SAM_WORKER_METRICS_PORT', '4609')),
        )
        logger.info(
            'sam_worker_pipeline_ready',
            stage_a_lpr_consumers=args.concurrency,
            stage_a_visible_consumers=gemma_visible_concurrency,
            stage_a_sam_consumers=args.concurrency,
            stage_b_consumers=gemma_concurrency,
            visible_chunk=VISIBLE_CHUNK,
            combined_chunk=COMBINED_CHUNK,
            in_q_max=in_q.maxsize,
            gemma_visible_q_max=gemma_visible_q.maxsize,
            sam_q_max=sam_q.maxsize,
            combined_q_max=combined_q.maxsize,
            out_q_max=out_q.maxsize,
        )

        # Wait for stop signal or producer-drained.
        await stop_event.wait()
        # Wait for producer to finish its current iteration.
        await prod_task
        # Drain Stage A.primary first: consumers push to either
        # combined_q or gemma_visible_q before exiting.
        for _ in range(args.concurrency):
            await in_q.put(None)
        await asyncio.gather(*stage_a_tasks, return_exceptions=True)
        # Drain Stage A.gemma_visible: pushes to sam_q or out_q.
        for _ in range(gemma_visible_concurrency):
            await gemma_visible_q.put(None)
        await asyncio.gather(*stage_a_visible_tasks, return_exceptions=True)
        # Drain Stage A.secondary: pushes to combined_q or out_q.
        for _ in range(args.concurrency):
            await sam_q.put(None)
        await asyncio.gather(*stage_a_sam_tasks, return_exceptions=True)
        # Drain Stage B: poison pills to combined_q, wait for consumers
        # to exit; they route their results to out_q.
        for _ in range(gemma_concurrency):
            await combined_q.put(None)
        await asyncio.gather(*stage_b_tasks, return_exceptions=True)
        # Now tell writer to flush + exit.
        await out_q.put(None)
        await writer_task
        # Stop the metrics task.
        metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await metrics_task
        # Shut the /metrics HTTP server down cleanly.
        with contextlib.suppress(Exception):
            await metrics_server_runner.cleanup()
    finally:
        await sam3.aclose()
        await gemma.aclose()
        await opensearch.close()
        await pool.close()

    elapsed_total = time.monotonic() - started_at
    logger.info(
        'sam_worker_done',
        processed=metrics['total_processed'],
        written=metrics['total_written'],
        elapsed_s=round(elapsed_total, 2),
        avg_cps=round(metrics['total_processed'] / max(elapsed_total, 1e-6), 2),
    )
    return 0

"""The detection worker's ``run()`` entry point: wiring and shutdown.

The pipeline's tasks live in ``flow``, ``stage_a``, ``stage_sam`` and
``stage_b``; shared state in ``pipeline``.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.curation.worker.crop_gate import CropGate
from scripts.curation.worker.fairness import FairnessScheduler, liveness_loop
from scripts.curation.worker.flow import metrics_reporter, producer, writer
from scripts.curation.worker.no_verdict import NoVerdictCounter, max_no_verdict_attempts
from scripts.curation.worker.pipeline import COMBINED_CHUNK, VISIBLE_CHUNK, PipelineContext
from scripts.curation.worker.stage_a import stage_a_consumer, stage_a_vlm_visible
from scripts.curation.worker.stage_b import stage_b_combined
from scripts.curation.worker.stage_sam import stage_a_sam_consumer
from src.core.logging import get_logger
from src.services.curation.worker_liveness import heartbeat_loop
from src.services.detection.cascade_detect.ocr_recognizer import PaddleOcrTextRecognizer
from src.services.detection.cascade_detect.region_detector import RegionDetector
from src.services.detection.profile_registry import get_active_region_profile


logger = get_logger('curation_worker')


if TYPE_CHECKING:
    import argparse

    from aiohttp import web

    from scripts.curation.worker.state import _ItemTask


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
    logger.info('region_worker_metrics_server_started', port=port)
    return runner


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
    # tests that monkeypatch `scripts.curation.region_worker_main.AsyncTritonPool`
    # (and friends) still intercept calls made from this split-out runner.
    from scripts.curation import region_worker_main as _wkr

    pool = _wkr.AsyncTritonPool(url=args.triton, pool_size=args.pool_size, max_concurrent=64)
    await pool.initialize()

    # LG-1: write RegionFields.embedding at flush time so a fresh install
    # doesn't depend solely on scripts/curation/backfill_region_embeddings.py
    # (the one-off catch-up for items ingested before this stage existed).
    # Gated: disabled entirely by OP_REGION_EMBED_ENABLED=false, or
    # auto-disabled with a warning if Triton doesn't report pe_image_encoder
    # READY at startup -- a missing/unloaded model must not spam a failure
    # log every flush.
    region_embed_pe = None
    if os.environ.get('OP_REGION_EMBED_ENABLED', 'true').strip().lower() not in (
        '0',
        'false',
        'no',
    ):
        from src.clients.pe_encoder import PE_IMAGE_MODEL, PEEncoder

        try:
            model_ready = bool(await pool.is_model_ready(PE_IMAGE_MODEL))
        except Exception as exc:
            logger.warning('region_embed_readiness_probe_failed', error=str(exc))
            model_ready = False
        if model_ready:
            region_embed_pe = PEEncoder(triton_pool=pool)
        else:
            logger.warning(
                'region_embed_disabled_model_not_ready',
                model=PE_IMAGE_MODEL,
                detail='region_box_embeddings will not be written this run',
            )

    # The VLM class catalog (prompt class list + name -> registry id) is
    # per project: every classifying call reads bound_class_catalog()
    # under the item's own binding, never a process-wide list.
    opensearch = _wkr.make_script_opensearch([args.opensearch])

    started_at = time.monotonic()
    sentinel = Path(args.pause_sentinel)

    from src.services.projects.registry import ProjectRegistry
    from src.services.projects.script_binding import only_project

    # Reuse the already-built (and, in tests, already-patched)
    # `opensearch` client above -- a second script_project_registry()
    # client would open its own real AsyncOpenSearch straight from
    # args.opensearch, bypassing whatever fake a caller/test installed
    # at _wkr.make_script_opensearch.
    registry = ProjectRegistry(lambda: opensearch)
    project_filter = getattr(args, 'project', None)

    class _WorkerProjects:
        """The registry as this worker sees it: active projects, narrowed
        to ``--project`` when given."""

        async def ensure_fresh(self) -> None:
            await registry.ensure_fresh()

        def active_projects(self) -> list[Any]:
            return only_project(registry.active_projects(), project_filter)

    project_registry = _WorkerProjects()
    fairness_scheduler = FairnessScheduler()

    # W2 sec 4.5, per-project (B1): one RegionRuntime per active project,
    # each fed by its OWN pinned ConfigStore -- alpha activating a
    # profile never touches beta's runtime because each slug's store
    # and holder entry are independent (projects_plan.md sec 11 W2
    # "runtimes[slug]"). The worker is multi-project by default (no
    # ``--project``); ``project_registry.active_projects()`` (already
    # narrowed to ``--project`` when given, via ``only_project``) is the
    # single source of truth for which slugs to hold a runtime for --
    # never the process's own (often unbound) context.
    from scripts.curation.worker.runtime import RuntimeHolder, maybe_hot_reload
    from src.config.project_context import bind_project
    from src.services.config_store.global_store import get_global_config_store
    from src.services.config_store.store import get_config_store as _get_config_store
    from src.services.labeling.vlm_endpoints import active_vlm_endpoint as _active_vlm_endpoint
    from src.services.labeling.vlm_prompts import active_prompt_pack as _active_prompt_pack

    runtime_holder = RuntimeHolder()
    # slug -> that project's pinned ConfigStore. Built the FIRST time
    # each project is seen, strictly before anything else for that
    # project resolves a store (B3: an accidental `mode='live'` store
    # from some other, earlier, default-mode `get_config_store()` call
    # would make the pinned design inert -- the store is created here,
    # pinned, before this project's first item is ever fetched).
    project_stores: dict[str, Any] = {}
    # W9: the deployment-wide VLM registry. ONE pinned store shared by every
    # project's sync, created here (before anything can create a live one) so
    # its snapshot only moves at a quiesce point like a project's own.
    registry_store = get_global_config_store(mode='pinned')
    # M1: `runtime:detection_worker:<host>` is written at startup, at
    # every swap, and at least every 60s (any_domain_plan.md sec 4.5
    # steps 2.6 / L803-805) -- throttled per project so N active
    # projects don't turn into N writes every single poll cycle.
    _runtime_doc_last_written: dict[str, float] = {}
    _RUNTIME_DOC_INTERVAL_S = 60.0
    import socket as _socket

    _hostname = _socket.gethostname()

    async def _sync_project_runtime(record: Any) -> None:
        """One project's hot-reload check (sec 4.5 step 2), run once per
        producer cycle for every active project. Binds ``record`` for
        the duration of the store refresh/build so every config read
        (``get_active_region_profile()``, ``active_prompt_pack()``,
        ``get_curation_config()``) resolves THIS project's own config,
        never whatever project happened to be bound before. Never
        suppresses ``ProjectNotBound`` or any other exception silently
        -- a failure here is logged and this project's existing runtime
        (if any) simply keeps running unchanged until the next cycle.
        """
        try:
            with bind_project(record):
                store = project_stores.get(record.slug)
                if store is None:
                    store = _get_config_store(mode='pinned')
                    project_stores[record.slug] = store
                rt = await maybe_hot_reload(
                    store=store,
                    registry=registry_store,
                    opensearch=opensearch,
                    holder=runtime_holder,
                    slug=record.slug,
                    pool=pool,
                    args=args,
                    queues=[in_q, vlm_visible_q, sam_q, combined_q, out_q],
                    get_active_profile=get_active_region_profile,
                    get_active_pack=_active_prompt_pack,
                    get_active_vlm=_active_vlm_endpoint,
                    region_detector_cls=RegionDetector,
                    ocr_recognizer_cls=PaddleOcrTextRecognizer,
                    segmenter_cls=_wkr.SegmenterClient,
                    build_vlm=_wkr.build_vlm_labeler,
                )
                if rt is None:
                    return
                last = _runtime_doc_last_written.get(record.slug, 0.0)
                if time.monotonic() - last < _RUNTIME_DOC_INTERVAL_S:
                    return
                from scripts.curation.worker.runtime import (
                    applied_config_revision,
                    upsert_project_runtime_doc,
                )

                await upsert_project_runtime_doc(
                    opensearch,
                    index=store.index,
                    hostname=_hostname,
                    project=record.slug,
                    runtime=rt,
                    config_revision=applied_config_revision(
                        store, registry_store, runtime_holder, record.slug
                    ),
                )
                _runtime_doc_last_written[record.slug] = time.monotonic()
        except Exception as exc:
            logger.warning('project_runtime_sync_failed', project=record.slug, error=str(exc))

    in_flight: set[str] = set()
    # crop_id -> owning project slug, for the per-project in-flight caps
    # and liveness counts (projects_plan.md §5.1).
    in_flight_owner: dict[str, str] = {}
    in_flight_lock = asyncio.Lock()
    # crop_id -> monotonic time the writer released it after a successful
    # write. A pending search that STARTED before that moment may carry the
    # pre-write (still pending) doc, so the producer drops those hits; a
    # search started after it sees the write (the bulk uses
    # refresh='wait_for'). Without this, a crop released between a
    # search's start and its response was re-queued and ran the whole
    # cascade a second time.
    released_at: dict[str, float] = {}

    # Pipeline (hybrid: ≤ 2 VLM round-trips per crop, with the cheap
    # visibility pre-filter shielding the slow segmenter GPU and the
    # heavy combined call from non-region crops):
    #
    #   in_q             -> Stage A.primary (pending_verify + primary-
    #                       detector Triton call)
    #                       splits into:
    #                         - has candidate         -> combined_q
    #                         - primary miss / secondary-shape -> vlm_visible_q
    #
    #   vlm_visible_q  -> Stage A.vlm_visible (batched yes/no
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
    vlm_visible_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=2000)
    sam_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=args.concurrency * 2)
    combined_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=2000)
    out_q: asyncio.Queue[_ItemTask | None] = asyncio.Queue(maxsize=args.concurrency * 8)
    # Number of Stage B (combined VLM) consumers. Each consumer drains
    # combined_q in chunks of COMBINED_CHUNK (6 per upstream call) so
    # the effective VLM concurrency = consumers x chunk = 16 x 6 = 96
    # in-flight. The removed visibility stage no longer competes for
    # VLM slots, so we can spend the full VLM budget on the combined
    # call (which subsumes both old calls).
    vlm_concurrency = int(os.environ.get('OP_REGION_WORKER_VLM_CONCURRENCY') or '16')
    # Visibility pre-filter: cheap yes/no, packed VISIBLE_CHUNK per call.
    # Default 8 consumers gives 8 x 6 = 48 in-flight calls at the
    # upstream VLM, well under the combined-call budget of 16 x 6 = 96.
    # The visible filter is fast (~2s/call) so it doesn't need as many
    # consumers as the heavier combined call.
    vlm_visible_concurrency = int(os.environ.get('OP_REGION_WORKER_VLM_VISIBLE_CONCURRENCY') or '8')

    metrics = {
        'total_processed': 0,
        'total_written': 0,
        'consecutive_empty_polls': 0,
        # vlm_visible_skipped: primary-miss / secondary-shape crops
        # that the visibility pre-filter short-circuited to
        # no_region_visible (no segmenter call, no combined call). The
        # point of this stage; bigger is better.
        'vlm_visible_skipped': 0,
        # vlm_visible_kept: crops that passed the filter and went on
        # to the secondary-segmenter stage. Together with skipped, lets
        # us compute the filter's skip rate at a glance.
        'vlm_visible_kept': 0,
        # visible_no_verdict: crops the visibility VLM call answered with
        # nothing (empty reply). Left pending for a retry, up to the
        # no-verdict cap.
        'visible_no_verdict': 0,
        # visible_no_verdict_cap_hits: crops that reached the cap there and
        # were sent on to detection (fail open).
        'visible_no_verdict_cap_hits': 0,
        # combined_bbox_wrong: the VLM said the region IS visible but
        # the proposed bbox was wrong. We write verify_rejected and do
        # NOT re-loop the segmenter (avoids re-introducing a 2nd VLM
        # call). If this counter climbs high, future work could add a
        # text-hint retry path.
        'combined_bbox_wrong': 0,
        # combined_parse_failure: per-crop entry missing or unparseable
        # in the batched response. Crop is dropped from in_flight so
        # the next producer poll re-fetches it, up to the no-verdict cap.
        'combined_parse_failure': 0,
        # combined_no_bbox_verdict: the VLM said a region is visible but
        # answered null / nothing on the candidate box. Left pending (no
        # write) for a retry, up to the no-verdict cap.
        'combined_no_bbox_verdict': 0,
        # combined_no_verdict_cap_hits: crops whose combined replies gave no
        # verdict (either kind above) on every allowed attempt; written
        # verify_rejected / verifier_no_verdict for human review.
        'combined_no_verdict_cap_hits': 0,
        # combined_no_region_visible: the VLM confirmed no region is
        # visible at all. Terminal write.
        'combined_no_region_visible': 0,
    }

    # A no-verdict reply (see no_verdict.py) is retried at most this many
    # times per item and stage; transport failures are never counted.
    no_verdict_cap = max_no_verdict_attempts()
    visible_no_verdict = NoVerdictCounter(no_verdict_cap)
    combined_no_verdict = NoVerdictCounter(no_verdict_cap)

    crop_gate = CropGate()
    ctx = PipelineContext(
        args=args,
        stop_event=stop_event,
        opensearch=opensearch,
        sentinel=sentinel,
        started_at=started_at,
        region_embed_pe=region_embed_pe,
        project_registry=project_registry,
        fairness_scheduler=fairness_scheduler,
        runtime_holder=runtime_holder,
        in_q=in_q,
        vlm_visible_q=vlm_visible_q,
        sam_q=sam_q,
        combined_q=combined_q,
        out_q=out_q,
        in_flight=in_flight,
        in_flight_owner=in_flight_owner,
        in_flight_lock=in_flight_lock,
        released_at=released_at,
        metrics=metrics,
        no_verdict_cap=no_verdict_cap,
        visible_no_verdict=visible_no_verdict,
        combined_no_verdict=combined_no_verdict,
        crop_gate=crop_gate,
    )

    logger.info(
        'region_worker_start_streaming',
        opensearch=args.opensearch,
        triton=args.triton,
        segmenter_url=args.segmenter_url,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        continuous=args.continuous,
        max_iterations=args.max_iterations,
        max_no_verdict_attempts=no_verdict_cap,
    )

    try:
        prod_task = asyncio.create_task(producer(ctx, _sync_project_runtime))
        # Stage A.primary pool: primary-detector + pending_verify
        # routing only. Sized to keep the primary-detector Triton
        # instances saturated; secondary-segmenter work has moved to
        # its own pool below.
        stage_a_tasks = [
            asyncio.create_task(stage_a_consumer(ctx, i)) for i in range(args.concurrency)
        ]
        # Stage A.vlm_visible pool: batched yes/no region-visibility
        # filter for primary-miss / secondary-shape crops. Sized
        # smaller than the combined pool because the visibility prompt
        # is cheap.
        stage_a_visible_tasks = [
            asyncio.create_task(stage_a_vlm_visible(ctx, i)) for i in range(vlm_visible_concurrency)
        ]
        # Stage A.secondary pool: secondary-segmenter run on every crop
        # the visibility filter said could contain a region. Sized to
        # keep the secondary-segmenter deployment saturated.
        stage_a_sam_tasks = [
            asyncio.create_task(stage_a_sam_consumer(ctx, i)) for i in range(args.concurrency)
        ]
        # Stage B pool: batched combined VLM call (COMBINED_CHUNK
        # crops per upstream call). vlm_concurrency x COMBINED_CHUNK
        # in-flight (16 x 6 = 96 by default). Each crop gets at most
        # ONE combined VLM round-trip here (plus at most ONE yes/no
        # call up in the visibility stage = ≤ 2 per crop total).
        stage_b_tasks = [
            asyncio.create_task(stage_b_combined(ctx, i)) for i in range(vlm_concurrency)
        ]
        writer_task = asyncio.create_task(writer(ctx))
        metrics_task = asyncio.create_task(metrics_reporter(ctx))
        liveness_task = asyncio.create_task(
            liveness_loop(
                project_registry.active_projects,
                lambda: collections.Counter(
                    owner for cid, owner in in_flight_owner.items() if cid in in_flight
                ),
                stop_event,
            )
        )
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(
                'detection_worker',
                lambda: {'producer': not prod_task.done(), 'writer': not writer_task.done()},
                stop_event,
            )
        )
        # Phase 4c: stand up an aiohttp /metrics endpoint inside the
        # worker process so Prometheus can scrape the stage-timing
        # histograms (the worker is not an HTTP server otherwise).
        # Port 4609 inside container, mapped 1:1 on the host so the
        # default prometheus scrape config can reach it.
        metrics_server_runner = await _start_metrics_http_server(
            port=int(os.environ.get('OP_REGION_WORKER_METRICS_PORT', '4609')),
        )
        logger.info(
            'region_worker_pipeline_ready',
            stage_a_detector_consumers=args.concurrency,
            stage_a_visible_consumers=vlm_visible_concurrency,
            stage_a_sam_consumers=args.concurrency,
            stage_b_consumers=vlm_concurrency,
            visible_chunk=VISIBLE_CHUNK,
            combined_chunk=COMBINED_CHUNK,
            in_q_max=in_q.maxsize,
            vlm_visible_q_max=vlm_visible_q.maxsize,
            sam_q_max=sam_q.maxsize,
            combined_q_max=combined_q.maxsize,
            out_q_max=out_q.maxsize,
        )

        # Wait for stop signal or producer-drained.
        await stop_event.wait()
        # Wait for producer to finish its current iteration.
        await prod_task
        # Drain Stage A.primary first: consumers push to either
        # combined_q or vlm_visible_q before exiting.
        for _ in range(args.concurrency):
            await in_q.put(None)
        await asyncio.gather(*stage_a_tasks, return_exceptions=True)
        # Drain Stage A.vlm_visible: pushes to sam_q or out_q.
        for _ in range(vlm_visible_concurrency):
            await vlm_visible_q.put(None)
        await asyncio.gather(*stage_a_visible_tasks, return_exceptions=True)
        # Drain Stage A.secondary: pushes to combined_q or out_q.
        for _ in range(args.concurrency):
            await sam_q.put(None)
        await asyncio.gather(*stage_a_sam_tasks, return_exceptions=True)
        # Drain Stage B: poison pills to combined_q, wait for consumers
        # to exit; they route their results to out_q.
        for _ in range(vlm_concurrency):
            await combined_q.put(None)
        await asyncio.gather(*stage_b_tasks, return_exceptions=True)
        # Now tell writer to flush + exit.
        await out_q.put(None)
        await writer_task
        # Stop the metrics + heartbeat tasks.
        metrics_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await metrics_task
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        liveness_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await liveness_task
        # Shut the /metrics HTTP server down cleanly.
        with contextlib.suppress(Exception):
            await metrics_server_runner.cleanup()
    finally:
        # Close every project's own segmenter/VLM clients (B1: no
        # single process-wide pair to close anymore).
        for _rt in runtime_holder.current.values():
            with contextlib.suppress(Exception):
                await _rt.segmenter.aclose()
            if _rt.vlm is not None:
                with contextlib.suppress(Exception):
                    await _rt.vlm.aclose()
        await opensearch.close()
        await pool.close()

    elapsed_total = time.monotonic() - started_at
    logger.info(
        'region_worker_done',
        processed=metrics['total_processed'],
        written=metrics['total_written'],
        elapsed_s=round(elapsed_total, 2),
        avg_cps=round(metrics['total_processed'] / max(elapsed_total, 1e-6), 2),
    )
    return 0

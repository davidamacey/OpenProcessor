"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/region_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

# ruff: noqa: E402
import asyncio
import collections
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
from src.services.curation.ingest_class_sources import (
    CLUSTER_MAJORITY_CLASS_SOURCE,
    classifier_class_sources,
)
from src.services.curation.metrics import (
    OP_STAGE_A_SEGMENTER_DURATION_SECONDS,
    OP_STAGE_A_VLM_VISIBLE_DURATION_SECONDS,
    OP_STAGE_B_VLM_VERIFY_DURATION_SECONDS,
    OP_STAGE_REGION_DETECTOR_DURATION_SECONDS,
)
from src.services.curation.region_boxes import RegionBox, boxes_write_fields, next_box_id
from src.services.curation.worker_liveness import heartbeat_loop
from src.services.detection.cascade_detect import (
    PaddleOcrTextRecognizer,
    RegionCandidate,
    RegionDetector,
    crop_norm_to_source_norm,
)
from src.services.detection.profile_registry import get_active_region_profile
from src.services.detection.region_candidates import select_region_candidates
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_labeler import CombinedCrop, RegionCrop


logger = get_logger('curation_worker')


from scripts.curation.worker import state
from scripts.curation.worker.bulk_writer import _bulk_update
from scripts.curation.worker.cascade import (
    SegmenterAllHostsDown,
    _resegment_from_text_hint,
    _source_to_crop,
)
from scripts.curation.worker.fairness import (
    FairnessScheduler,
    fetch_pending_multi_project,
    is_project_paused,
    liveness_loop,
    write_liveness,
)
from scripts.curation.worker.no_verdict import NoVerdictCounter, max_no_verdict_attempts
from scripts.curation.worker.region_embed_stage import embed_written_regions
from scripts.curation.worker.region_text_stage import (
    _box_with_resolved_text,
    accept_without_vlm,
    apply_region_text,
    apply_text_hint_fallback,
    item_text_fields,
    read_item_lines,
)
from scripts.curation.worker.state import (
    _PENDING_DETECTION_ALIASES,
    _PENDING_VERIFICATION_ALIASES,
    _TERMINAL_STATUSES,
    RegionProfileNotConfiguredError,
    _crop_jpeg_for_task,
    _is_secondary_shape,
    _ItemTask,
    _wait_for_sentinel_clear,
    bind_task_project,
    bound_class_catalog,
    unreadable_crop_update,
)
from scripts.curation.worker.verify import (
    _SKIP_VLM_VERIFY_SECONDARY_SCORE,
    TaskBoxInput,
    _bbox_shape_is_plausible,
    _combined_class_update,
    verdicts_to_boxes,
)
from src.config.region_rejection import REJECT_REASON_VERIFIER
from src.config.region_source import (
    CANDIDATE_DETECTOR,
    CANDIDATE_DETECTOR_EXISTING,
    CANDIDATE_SEGMENTER,
    CANDIDATE_SEGMENTER_TEXT_HINT,
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
    logger.info('region_worker_metrics_server_started', port=port)
    return runner


# High-conf primary-classifier cohort skips classification in the
# combined call (the caller already has a trusted class). Same
# threshold as the legacy cascade's combined-cohort gate
# (combined.py: _CLASSIFIER_LOW_CONF_THRESHOLD).
_CLASSIFIER_HIGH_CONF_THRESHOLD = 0.80

# How long the producer remembers a released crop. Only has to outlive the
# slowest single pending search.
_RELEASED_AT_TTL_S = 300.0


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
        t.class_source in (classifier_class_sources() | {CLUSTER_MAJORITY_CLASS_SOURCE})
        and t.class_confidence >= _CLASSIFIER_HIGH_CONF_THRESHOLD
    )


# =============================================================================
# W8 multi-box candidate wiring helpers (module-level so they're directly
# unit-testable, same rationale as ``_should_classify``).
# =============================================================================


def _select_candidates(
    raw: list[Any],
    *,
    profile: Any,
    item_bbox_norm: tuple[float, float, float, float],
    detector: str,
    detector_version: str,
    source: str,
    min_score: float = 0.0,
) -> list[TaskBoxInput]:
    """Floor/NMS/cap ``raw`` (a detector/segmenter leg's candidate list)
    then wrap the selection as :class:`TaskBoxInput`, source-frame bbox
    projected via ``item_bbox_norm``. Every selected candidate is fresh
    (``box_id=None`` -- ``verdicts_to_boxes`` mints one).
    """
    sel = select_region_candidates(
        raw,
        min_score=min_score,
        iou=profile.region_nms_iou,
        max_n=profile.region_max_candidates,
    )
    return [
        TaskBoxInput(
            bbox_in_crop=c.bbox_norm,
            bbox_in_source=crop_norm_to_source_norm(c.bbox_norm, item_bbox_norm),
            score=c.score,
            detector=detector,
            detector_version=detector_version,
            source=source,
        )
        for c in sel.selected
    ]


def _box_list_doc(
    t: _ItemTask,
    boxes: list[RegionBox],
    status: RegionStatus,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The box-list ``update_doc`` every W8 terminal write shares: item
    status + ``boxes_write_fields`` (region_boxes / count / rejected_count
    / max_score / revision / box_seq) + the detector chain. ``current_src``
    for the revision/seq bump comes from this task's own read-time
    snapshot (``t.region_revision`` / ``t.region_box_seq``)."""
    F = get_region_fields()
    current_src = {F.revision: t.region_revision, F.box_seq: t.region_box_seq}
    doc: dict[str, Any] = {
        F.status: status,
        **boxes_write_fields(boxes, current_src=current_src),
    }
    if t.detection_trace:
        doc[F.detector_chain] = list(t.detection_trace)
    if extra:
        doc.update(extra)
    return doc


def _sync_singular_candidate(t: _ItemTask) -> None:
    """Mirror ``t.candidates[0]`` onto the legacy singular ``candidate_*``
    fields.

    Kept for the two consumers that are still deliberately single-box:
    ``accept_without_vlm`` (no VLM configured -- nothing can adjudicate
    between multiple candidates) and ``embed_written_regions`` (item-level
    ``RegionFields.embedding``; per-box embeddings are W8c scope, not this
    pass). A no-op when ``t.candidates`` is empty.
    """
    if not t.candidates:
        return
    best = t.candidates[0]
    t.candidate_in_crop = best.bbox_in_crop
    t.candidate_in_source = best.bbox_in_source
    t.candidate_score = best.score
    t.candidate_source = best.source


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
                detail='region_embedding will not be written this run',
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
    from src.services.config_store.store import get_config_store as _get_config_store
    from src.services.labeling.vlm_prompts import active_prompt_pack as _active_prompt_pack

    runtime_holder = RuntimeHolder()
    # slug -> that project's pinned ConfigStore. Built the FIRST time
    # each project is seen, strictly before anything else for that
    # project resolves a store (B3: an accidental `mode='live'` store
    # from some other, earlier, default-mode `get_config_store()` call
    # would make the pinned design inert -- the store is created here,
    # pinned, before this project's first item is ever fetched).
    project_stores: dict[str, Any] = {}
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
                    opensearch=opensearch,
                    holder=runtime_holder,
                    slug=record.slug,
                    pool=pool,
                    args=args,
                    queues=[in_q, vlm_visible_q, sam_q, combined_q, out_q],
                    get_active_profile=get_active_region_profile,
                    get_active_pack=_active_prompt_pack,
                    region_detector_cls=RegionDetector,
                    ocr_recognizer_cls=PaddleOcrTextRecognizer,
                    segmenter_cls=_wkr.SegmenterClient,
                    vlm_cls=_wkr.VlmLabeler,
                )
                if rt is None:
                    return
                last = _runtime_doc_last_written.get(record.slug, 0.0)
                if time.monotonic() - last < _RUNTIME_DOC_INTERVAL_S:
                    return
                from scripts.curation.worker.runtime import upsert_project_runtime_doc

                await upsert_project_runtime_doc(
                    opensearch,
                    index=store.index,
                    hostname=_hostname,
                    project=record.slug,
                    runtime=rt,
                    config_revision=store.current.config_revision,
                )
                _runtime_doc_last_written[record.slug] = time.monotonic()
        except Exception as exc:
            logger.warning('project_runtime_sync_failed', project=record.slug, error=str(exc))

    def _rt_for(t: _ItemTask) -> Any:
        """The runtime this item's own project is currently on. Never a
        process-wide default -- a project with no runtime yet (no
        profile configured anywhere, or not synced this cycle) raises,
        which every stage's existing `except Exception` handler turns
        into "drop from in_flight, write nothing" (M2's no-profile-wait
        semantics, applied per project instead of per process)."""
        rt = runtime_holder.get(t.project.slug)
        if rt is None:
            msg = f"no region runtime built yet for project '{t.project.slug}'"
            raise RegionProfileNotConfiguredError(msg)
        return rt

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
    VLM_CHUNK_DRAIN_TIMEOUT = 0.10

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

    async def producer() -> None:
        """Fetch eligible crops and queue task DESCRIPTORS only.

        In-flight crops are now excluded server-side (``must_not
        ids``), so the fetch no longer needs to over-fetch
        ``batch_size + in_flight_count`` and then filter in Python — the
        old bug this over-fetch fixed (batch_size=96, in_flight=132, the
        96 oldest all in-flight, fresh=0) can't recur when the query
        itself already excludes in-flight ids.
        """
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
                await _sync_project_runtime(_record)

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

    async def stage_a_consumer(consumer_id: int) -> None:
        """Stage A.primary: load JPEG + primary/pending_verify routing only.

        Never calls the VLM or the secondary segmenter directly. Routes:
          - pending_verify (existing primary-detector candidate) -> combined_q
          - pending + non-secondary-shape with primary hit       -> combined_q
          - pending + secondary-shape OR primary miss             -> vlm_visible_q
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
            # §5.1.5: bind this item's own project for the duration of
            # its processing on this consumer, so every downstream
            # config/OpenSearch/registry read below resolves against
            # the item's project, not whatever project a previous item
            # on this consumer happened to be. No-op (stays unbound)
            # for legacy single-project tasks with ``project is None``.
            bind_task_project(t)
            # Bind request_id so every structlog event in this iteration
            # carries it (Phase 4a). Cleared in finally so the next task
            # on this consumer task doesn't inherit the previous id.
            structlog.contextvars.bind_contextvars(request_id=t.request_id)
            try:
                # B1: this item's OWN project's runtime, resolved fresh
                # every item -- never a process-wide detector/segmenter/
                # vlm/profile. Raises (caught below, dropped from
                # in_flight, no write) when this project has no runtime
                # yet (M2: no profile configured anywhere for it).
                rt = _rt_for(t)
                # Load JPEG (parallel HDD reads across all consumers).
                if t.crop_jpeg is None:
                    t.crop_jpeg = await asyncio.to_thread(
                        _crop_jpeg_for_task,
                        t.crop_id,
                        t.image_path,
                        t.item_bbox_norm,
                    )
                if t.crop_jpeg is None:
                    t.update_doc = unreadable_crop_update(t)
                    await out_q.put(t)
                    in_q.task_done()
                    continue
                if t.region_status in _TERMINAL_STATUSES:
                    async with in_flight_lock:
                        in_flight.discard(t.crop_id)
                    in_q.task_done()
                    continue

                # One OCR read of the item crop per pass: stored as the
                # item's searchable text and reused by the text-hint step.
                if rt.item_text_enabled:
                    t.item_ocr_lines = await read_item_lines(
                        rt.ocr_recognizer, t.crop_jpeg, t.crop_id
                    )
                    t.item_text_update = item_text_fields(
                        t.item_ocr_lines, min_confidence=rt.item_text_min_conf
                    )

                is_secondary = _is_secondary_shape(t)

                # Path 1: pending_verify — already has a primary-detector
                # candidate; straight to combined VLM call (no
                # detection needed). W8: a single stored candidate is
                # still a list of one (select_region_candidates applies
                # the same floor/cap uniformly, even to N=1).
                if (
                    t.region_status in _PENDING_VERIFICATION_ALIASES
                    and t.detector_region_in_source is not None
                ):
                    cand_in_crop = _source_to_crop(t.detector_region_in_source, t.item_bbox_norm)
                    t.candidates = _select_candidates(
                        [
                            RegionCandidate(
                                bbox_norm=cand_in_crop,
                                score=t.detector_score,
                                source=CANDIDATE_DETECTOR_EXISTING,
                            )
                        ],
                        profile=rt.profile,
                        item_bbox_norm=t.item_bbox_norm,
                        detector=rt.profile.detector_model,
                        detector_version=rt.profile.detector_version,
                        source=CANDIDATE_DETECTOR_EXISTING,
                    )
                    _sync_singular_candidate(t)
                    if t.candidates:
                        if rt.vlm_available:
                            await combined_q.put(t)
                        else:
                            await accept_without_vlm(
                                t, ocr=rt.ocr_recognizer, profile=rt.profile, rules=rt.text_rules
                            )
                            await out_q.put(t)
                    else:
                        t.update_doc = _box_list_doc(t, [], RegionStatus.NO_REGION_BOX)
                        await out_q.put(t)
                    in_q.task_done()
                    continue

                # Path 2: pending + non-secondary-shape — try the
                # primary detector first (fast Triton call). A profile with
                # no detector_model has no detector leg: straight to Path 3,
                # with nothing on the trace.
                if (
                    t.region_status in _PENDING_DETECTION_ALIASES
                    and not is_secondary
                    and rt.profile.detector_model
                ):
                    _detector_t0 = time.monotonic()
                    try:
                        detector_results = await rt.detector.detect_batch_multi([t.crop_jpeg])
                    except Exception:
                        OP_STAGE_REGION_DETECTOR_DURATION_SECONDS.labels(outcome='error').observe(
                            time.monotonic() - _detector_t0
                        )
                        raise
                    raw_cands = detector_results[0] if detector_results else []
                    OP_STAGE_REGION_DETECTOR_DURATION_SECONDS.labels(
                        outcome='hit' if raw_cands else 'miss'
                    ).observe(time.monotonic() - _detector_t0)
                    if raw_cands:
                        t.detection_trace.append(f'{rt.profile.detector_model}:hit')
                        t.candidates = _select_candidates(
                            raw_cands,
                            profile=rt.profile,
                            item_bbox_norm=t.item_bbox_norm,
                            detector=rt.profile.detector_model,
                            detector_version=rt.profile.detector_version,
                            source=CANDIDATE_DETECTOR,
                            min_score=rt.detector.confidence_floor,
                        )
                        _sync_singular_candidate(t)
                    else:
                        # Recorded so the blind-spot training cohort
                        # (``<detector>:miss`` + segmenter hit) can find it.
                        t.detection_trace.append(f'{rt.profile.detector_model}:miss')
                    if t.candidates:
                        if rt.vlm_available:
                            await combined_q.put(t)
                        else:
                            await accept_without_vlm(
                                t, ocr=rt.ocr_recognizer, profile=rt.profile, rules=rt.text_rules
                            )
                            await out_q.put(t)
                        in_q.task_done()
                        continue
                    # No candidates (detector miss, or every raw candidate
                    # was floored/NMS'd away) -- fall through to Path 3.

                # Path 3: secondary-shape pending OR non-secondary with
                # no primary hit. Hand off to the visibility
                # pre-filter — VLM yes/no decides whether the slow
                # segmenter + combined path is even worth it. Fails
                # OPEN on parse errors so a flaky VLM never silently
                # drops a real region. No VLM: straight to the segmenter.
                await (vlm_visible_q if rt.vlm_available else sam_q).put(t)
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
        carry: list[_ItemTask],
    ) -> tuple[list[_ItemTask], bool]:
        """Pull up to ``chunk_size`` tasks of ONE project from ``q``.

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

        A batched VLM call runs under one project binding, so a chunk
        never mixes projects: the first task of another project ends the
        chunk and is parked in ``carry`` (owned by the calling consumer),
        which starts that consumer's next chunk. A parked task never
        coexists with a consumed poison pill, so shutdown cannot strand it.
        """

        tasks: list[_ItemTask] = []
        first = carry.pop() if carry else None
        if first is None:
            first = await q.get()
            q.task_done()
            if first is None:
                return tasks, True
        tasks.append(first)

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
            q.task_done()
            if t.project != first.project:
                carry.append(t)
                break
            tasks.append(t)
        return tasks, poisoned

    async def stage_a_vlm_visible(consumer_id: int) -> None:
        """Stage A.vlm_visible: batched yes/no region-visibility filter.

        Drains ``vlm_visible_q`` in chunks of ``VISIBLE_CHUNK`` and
        asks the VLM "is a region of interest visible at all?" per
        crop. Crops that come back ``False`` short-circuit to
        ``no_region_visible`` without ever touching the secondary
        segmenter or the combined call — that's the entire point of
        this stage. Crops that come back ``True`` (or that the parser
        fails open on) advance to ``sam_q``.

        Fail-OPEN semantics: any RPC error and any per-entry parse
        failure is treated as ``visible=True`` so we never silently bin
        a real region when the VLM is flaky. The cost is one extra
        segmenter round-trip on those crops — the existing pipeline is
        the safety net. An empty response is no verdict: those crops are
        left pending and retried (never stamped "no region visible"), up
        to the no-verdict cap; at the cap they fail open to ``sam_q``.
        """

        carry: list[_ItemTask] = []
        while True:
            chunk, poisoned = await _drain_chunk(
                vlm_visible_q,
                chunk_size=VISIBLE_CHUNK,
                drain_timeout=VLM_CHUNK_DRAIN_TIMEOUT,
                carry=carry,
            )
            if chunk:
                # _drain_chunk returns single-project chunks, so one
                # binding + one runtime resolution covers the whole
                # batched call. A project with no runtime yet drops the
                # whole chunk from in_flight (retried next poll) rather
                # than crashing this consumer task.
                bind_task_project(chunk[0])
                try:
                    rt = _rt_for(chunk[0])
                except RegionProfileNotConfiguredError as exc:
                    logger.warning(
                        'stage_a_vlm_visible_no_runtime',
                        project=chunk[0].project.slug,
                        error=str(exc),
                    )
                    async with in_flight_lock:
                        for t in chunk:
                            in_flight.discard(t.crop_id)
                    if poisoned:
                        return
                    continue
                batch_request_ids = [t.request_id for t in chunk]
                region_crops: list[RegionCrop] = []
                bad_indices: list[int] = []
                for i, t in enumerate(chunk):
                    if t.crop_jpeg is None:
                        bad_indices.append(i)
                        continue
                    region_crops.append(RegionCrop(crop_id=t.crop_id, jpeg_bytes=t.crop_jpeg))

                verdicts: dict[str, bool] = {}
                if region_crops:
                    _vis_t0 = time.monotonic()
                    try:
                        if rt.vlm is None:
                            msg = 'visibility stage fed without a VLM'
                            raise RuntimeError(msg)
                        verdicts = await rt.vlm.region_visible_batch(region_crops)
                        # Minor 5 (W2 review): only a write this call actually
                        # informed gets stamped `vlm_prompt_pack` downstream.
                        for _t in chunk:
                            if _t.crop_jpeg is not None:
                                _t.vlm_called = True
                        OP_STAGE_A_VLM_VISIBLE_DURATION_SECONDS.labels(outcome='ok').observe(
                            time.monotonic() - _vis_t0
                        )
                    except Exception as exc:
                        OP_STAGE_A_VLM_VISIBLE_DURATION_SECONDS.labels(outcome='error').observe(
                            time.monotonic() - _vis_t0
                        )
                        logger.warning(
                            'stage_a_vlm_visible_failed',
                            consumer_id=consumer_id,
                            chunk_size=len(region_crops),
                            request_ids=batch_request_ids,
                            error=str(exc),
                        )
                        # Fail-OPEN: pretend everyone is visible so the
                        # secondary segmenter still gets a crack at them.
                        verdicts = {c.crop_id: True for c in region_crops}

                F = get_region_fields()
                for i, t in enumerate(chunk):
                    structlog.contextvars.bind_contextvars(request_id=t.request_id)
                    try:
                        if i in bad_indices:
                            t.update_doc = unreadable_crop_update(t)
                            await out_q.put(t)
                            continue
                        is_visible = verdicts.get(t.crop_id)
                        if is_visible is None:
                            # No verdict (the VLM answered the chunk with
                            # nothing): not a "no region visible". Leave the
                            # item pending -- drop it from in_flight so the
                            # next producer poll retries it -- until the cap.
                            metrics['visible_no_verdict'] += 1
                            if not visible_no_verdict.record(t.crop_id):
                                async with in_flight_lock:
                                    in_flight.discard(t.crop_id)
                                continue
                            # Cap reached: fail open, like the stage's other
                            # no-answer paths -- detection decides.
                            metrics['visible_no_verdict_cap_hits'] += 1
                            logger.warning(
                                'region_worker_no_verdict_cap',
                                stage='visibility',
                                crop_id=t.crop_id,
                                attempts=no_verdict_cap,
                            )
                            t.detection_trace.append('vlm_visible:no_verdict')
                            await sam_q.put(t)
                            continue
                        visible_no_verdict.clear(t.crop_id)
                        if is_visible:
                            metrics['vlm_visible_kept'] += 1
                            t.detection_trace.append('vlm_visible:yes')
                            await sam_q.put(t)
                        else:
                            metrics['vlm_visible_skipped'] += 1
                            t.detection_trace.append('vlm_visible:no')
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
        came through ``stage_a_vlm_visible``). High-confidence
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
            bind_task_project(t)
            structlog.contextvars.bind_contextvars(request_id=t.request_id)
            try:
                rt = _rt_for(t)
                if t.crop_jpeg is None:
                    t.update_doc = unreadable_crop_update(t)
                    await out_q.put(t)
                    sam_q.task_done()
                    continue

                _sam_t0 = time.monotonic()
                try:
                    raw_sam_cands = await rt.segmenter.segment_multi(t.crop_jpeg)
                except SegmenterAllHostsDown as exc:
                    # Infrastructure failure (every secondary-segmenter
                    # host UNHEALTHY). Do NOT mark the crop terminal —
                    # leave region_status unchanged so it stays in
                    # pending_detection for the next cascade pass once
                    # a host recovers. Drop from in_flight + sleep so
                    # the producer can re-fetch and we don't spin a hot
                    # loop while every host is down.
                    OP_STAGE_A_SEGMENTER_DURATION_SECONDS.labels(outcome='error').observe(
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
                    OP_STAGE_A_SEGMENTER_DURATION_SECONDS.labels(outcome='error').observe(
                        time.monotonic() - _sam_t0
                    )
                    raise
                _sam_elapsed = time.monotonic() - _sam_t0
                OP_STAGE_A_SEGMENTER_DURATION_SECONDS.labels(
                    outcome='hit' if raw_sam_cands else 'miss'
                ).observe(_sam_elapsed)
                logger.info(
                    'stage_a_sam_took_ms',
                    crop_id=t.crop_id,
                    ms=round(_sam_elapsed * 1000.0, 2),
                    hit=bool(raw_sam_cands),
                )
                sam_selected = _select_candidates(
                    raw_sam_cands,
                    profile=rt.profile,
                    item_bbox_norm=t.item_bbox_norm,
                    detector=rt.profile.segmenter_name,
                    detector_version=rt.profile.segmenter_version,
                    source=CANDIDATE_SEGMENTER,
                )
                if sam_selected:
                    # High-conf-skip: bypass VLM verify entirely, but only
                    # when there is exactly ONE candidate to auto-accept
                    # -- with N>1 candidates the VLM is the adjudicator
                    # (that's the point of offering more than one), so the
                    # skip shortcut never applies to a multi-candidate set.
                    top = sam_selected[0]
                    if (
                        len(sam_selected) == 1
                        and top.score >= _SKIP_VLM_VERIFY_SECONDARY_SCORE
                        and _bbox_shape_is_plausible(top.bbox_in_crop)
                    ):
                        t.detection_trace.append(f'{rt.profile.segmenter_name}:hit')
                        t.detection_trace.append(f'{rt.profile.segmenter_name}:skip_vlm_verify')
                        box = RegionBox(
                            box_id=next_box_id([], seq=t.region_box_seq),
                            bbox_norm=top.bbox_in_source,
                            state='accepted',
                            score=top.score,
                            detector=rt.profile.segmenter_name,
                            detector_version=rt.profile.segmenter_version,
                            source=CANDIDATE_SEGMENTER,
                        )
                        text_doc: dict[str, Any] = {}
                        await apply_region_text(
                            text_doc,
                            ocr=rt.ocr_recognizer,
                            crop_jpeg=t.crop_jpeg,
                            region_in_crop=top.bbox_in_crop,
                            profile=rt.profile,
                            crop_id=t.crop_id,
                            vlm_text=None,
                            vlm_confidence=None,
                            vlm_available=rt.vlm_available,
                            rules=rt.text_rules,
                        )
                        box = _box_with_resolved_text(box, text_doc, F)
                        t.update_doc = _box_list_doc(
                            t, [box], RegionStatus.DETECTED, extra={F.skip_verify: True}
                        )
                        await out_q.put(t)
                        sam_q.task_done()
                        continue
                    # Else: queue the secondary-segmenter candidate(s) for
                    # a combined VLM call.
                    t.detection_trace.append(f'{rt.profile.segmenter_name}:hit')
                    t.candidates = sam_selected
                    _sync_singular_candidate(t)
                    if rt.vlm_available:
                        await combined_q.put(t)
                    else:
                        await accept_without_vlm(
                            t, ocr=rt.ocr_recognizer, profile=rt.profile, rules=rt.text_rules
                        )
                        await out_q.put(t)
                    sam_q.task_done()
                    continue

                if not rt.text_hint_on:
                    # No text-hint re-pass: the segmenter's miss is final.
                    if rt.segmenter.enabled:
                        t.detection_trace.append(f'{rt.profile.segmenter_name}:miss')
                else:
                    # Secondary segmenter missed globally; re-prompt it
                    # with a tight sub-crop around an OCR text hint. The
                    # OCR-detection bbox is no longer trusted; the
                    # segmenter produces the final geometry. OCR text
                    # rides along for storage.
                    if t.item_ocr_lines is not None:
                        ocr_regions = rt.ocr_recognizer.regions_from_lines(t.item_ocr_lines)
                    else:
                        try:
                            ocr_regions = await rt.ocr_recognizer.detect_regions(t.crop_jpeg)
                        except Exception as exc:
                            logger.warning(
                                'text_hint_ocr_failed', crop_id=t.crop_id, error=str(exc)
                            )
                            ocr_regions = []
                    ocr_pick = (
                        rt.ocr_recognizer.pick_best_text_region(ocr_regions)
                        if ocr_regions
                        else None
                    )
                    if ocr_pick is not None:
                        t.detection_trace.append(f'{rt.profile.ocr_rec_model}:text_hint:hit')
                        sub_cand, _sub_box = await _resegment_from_text_hint(
                            t.crop_jpeg, ocr_pick.bbox_norm, rt.segmenter
                        )
                        if sub_cand is not None:
                            t.candidates = [
                                TaskBoxInput(
                                    bbox_in_crop=sub_cand.bbox_norm,
                                    bbox_in_source=crop_norm_to_source_norm(
                                        sub_cand.bbox_norm, t.item_bbox_norm
                                    ),
                                    score=sub_cand.score,
                                    detector=rt.profile.segmenter_name,
                                    detector_version=rt.profile.segmenter_version,
                                    source=CANDIDATE_SEGMENTER_TEXT_HINT,
                                    hint_text=ocr_pick.text,
                                    hint_text_confidence=ocr_pick.rec_score,
                                )
                            ]
                            _sync_singular_candidate(t)
                            t.candidate_text = ocr_pick.text
                            t.candidate_text_confidence = ocr_pick.rec_score
                            if rt.vlm_available:
                                await combined_q.put(t)
                            else:
                                await accept_without_vlm(
                                    t,
                                    ocr=rt.ocr_recognizer,
                                    profile=rt.profile,
                                    rules=rt.text_rules,
                                )
                                await out_q.put(t)
                            sam_q.task_done()
                            continue
                        t.detection_trace.append(f'{rt.profile.segmenter_name}:text_hint:miss')
                    elif ocr_regions:
                        t.detection_trace.append(
                            f'{rt.profile.ocr_rec_model}:text_hint:no_region_shape'
                        )
                    else:
                        t.detection_trace.append(f'{rt.profile.ocr_rec_model}:text_hint:miss')

                # Nothing found by any detector → no_region_box.
                t.update_doc = _box_list_doc(t, [], RegionStatus.NO_REGION_BOX)
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

        Replaces the legacy vlm_visible + vlm_verify two-call
        cascade with ONE VLM round-trip per crop. W8: every candidate box
        selected for this item (``t.candidates``, 1..N) is drawn as a
        numbered overlay on the parent item JPEG before sending so the
        VLM confirms each one visually in the same call that classifies
        the item and reads the region text.

        Per-crop branches on the reply:
          - ``region_visible=False`` -> write 'no_region_visible' + class
            fields (checked first -- independent of any per-box verdict).
          - every box verdict ``bbox_correct is None`` (no candidate got
            an answer at all -- includes a missing/unparseable reply) ->
            no verdict: no write, the item stays pending for a retry, up
            to the no-verdict cap; then every box resolves ``rejected``
            / ``verifier_no_verdict`` (:func:`verdicts_to_boxes`
            ``force_resolve``).
          - otherwise -> :func:`verdicts_to_boxes` resolves each box
            (``accepted`` / ``rejected`` + reason) and the item status
            follows W8.7's precedence (accepted > false_positive >
            proposed > rejected > empty). ``combined_bbox_wrong`` counts
            crops where at least one candidate's box was visible-
            elsewhere-rejected.
          - the call itself failed (transport) -> every crop in the chunk
            is retried, never counted toward the cap.
        """

        F = get_region_fields()
        carry: list[_ItemTask] = []
        while True:
            chunk, poisoned = await _drain_chunk(
                combined_q,
                chunk_size=COMBINED_CHUNK,
                drain_timeout=VLM_CHUNK_DRAIN_TIMEOUT,
                carry=carry,
            )
            if chunk:
                # Single-project chunk (see _drain_chunk): classify
                # against that project's own registry.
                bind_task_project(chunk[0])
                try:
                    rt = _rt_for(chunk[0])
                except RegionProfileNotConfiguredError as exc:
                    logger.warning(
                        'stage_b_combined_no_runtime',
                        project=chunk[0].project.slug,
                        error=str(exc),
                    )
                    async with in_flight_lock:
                        for t in chunk:
                            in_flight.discard(t.crop_id)
                    if poisoned:
                        return
                    continue
                batch_request_ids = [t.request_id for t in chunk]
                class_names, name_to_id = bound_class_catalog()
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
                            region_bboxes_norm=[c.bbox_in_crop for c in t.candidates],
                            classify=_should_classify(t, registry_loaded=registry_loaded),
                        )
                    )

                replies_by_id: dict[str, Any] = {}
                if combined_crops:
                    _vlm_t0 = time.monotonic()
                    try:
                        if rt.vlm is None:
                            msg = 'combined stage fed without a VLM'
                            raise RuntimeError(msg)
                        replies_by_id = await rt.vlm.label_combined_batch(
                            combined_crops,
                            class_names=class_names or None,
                        )
                        # Minor 5 (W2 review): only a write this call actually
                        # informed gets stamped `vlm_prompt_pack` downstream.
                        for _t in chunk:
                            if _t.crop_jpeg is not None:
                                _t.vlm_called = True
                        _vlm_elapsed = time.monotonic() - _vlm_t0
                        OP_STAGE_B_VLM_VERIFY_DURATION_SECONDS.labels(outcome='ok').observe(
                            _vlm_elapsed
                        )
                        _vlm_ms = round(_vlm_elapsed * 1000.0, 2)
                        logger.info(
                            'stage_b_combined_took_ms',
                            consumer_id=consumer_id,
                            chunk_size=len(combined_crops),
                            request_ids=batch_request_ids,
                            ms=_vlm_ms,
                            per_crop_ms=round(_vlm_ms / max(1, len(combined_crops)), 2),
                        )
                    except Exception as exc:
                        OP_STAGE_B_VLM_VERIFY_DURATION_SECONDS.labels(outcome='error').observe(
                            time.monotonic() - _vlm_t0
                        )
                        logger.warning(
                            'stage_b_combined_failed',
                            consumer_id=consumer_id,
                            chunk_size=len(combined_crops),
                            request_ids=batch_request_ids,
                            error=str(exc),
                        )
                        # Transport failure (no reply at all): drop everyone
                        # in this chunk from in_flight so the producer
                        # re-fetches; don't write any update_doc so OS
                        # state stays unchanged. Never counted toward the
                        # no-verdict cap -- an outage must not turn into
                        # terminal writes.
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
                            t.update_doc = unreadable_crop_update(t)
                            await out_q.put(t)
                            continue
                        reply = replies_by_id.get(t.crop_id)

                        # Decide per-crop class-field side: only honor
                        # the VLM's class when we asked it to classify.
                        effective_class_names = (
                            class_names
                            if _should_classify(t, registry_loaded=registry_loaded)
                            else None
                        )

                        # Canonical detector name + version for provenance
                        # (the whole item's candidates share one leg per
                        # pass -- Path 1/2/segmenter never mix in the same
                        # combined_q push).
                        actor = t.candidates[0].detector if t.candidates else 'unknown'
                        # The candidate's detector gets exactly one ``:hit``
                        # (Stage A records it for fresh detections; an
                        # ingest-time box awaiting verification has none).
                        if f'{actor}:hit' not in t.detection_trace:
                            t.detection_trace.append(f'{actor}:hit')

                        if reply is not None and not reply.region_visible:
                            # region_visible=False — no region in this crop,
                            # regardless of any per-box verdict.
                            metrics['combined_no_region_visible'] += 1
                            t.detection_trace.append(f'{actor}:combined_no_region_visible')
                            t.update_doc = _box_list_doc(
                                t,
                                [],
                                RegionStatus.NO_REGION_VISIBLE,
                                extra=_combined_class_update(
                                    reply, effective_class_names, name_to_id=name_to_id
                                ),
                            )
                            combined_no_verdict.clear(t.crop_id)
                            await out_q.put(t)
                            continue

                        no_verdicts = [
                            VlmBoxVerdict(box=i, bbox_correct=None, confidence=None)
                            for i in range(1, len(t.candidates) + 1)
                        ]
                        boxes, status, extra = verdicts_to_boxes(
                            t.candidates,
                            reply.region_boxes if reply is not None else no_verdicts,
                            seq=t.region_box_seq,
                            item_bbox_norm=t.item_bbox_norm,
                        )
                        if extra.get('no_verdict'):
                            # No candidate got any verdict at all -- the
                            # entry is missing/unparseable, or the VLM saw
                            # a region but answered null/nothing on every
                            # box. Not a reject: leave the item pending --
                            # drop it from in_flight so the next producer
                            # poll retries it -- until the no-verdict cap.
                            if reply is None:
                                metrics['combined_parse_failure'] += 1
                                logger.warning(
                                    'stage_b_combined_parse_failure',
                                    crop_id=t.crop_id,
                                    request_id=t.request_id,
                                )
                            else:
                                metrics['combined_no_bbox_verdict'] += 1
                                logger.info(
                                    'stage_b_combined_no_bbox_verdict',
                                    crop_id=t.crop_id,
                                    request_id=t.request_id,
                                )
                            if not combined_no_verdict.record(t.crop_id):
                                async with in_flight_lock:
                                    in_flight.discard(t.crop_id)
                                continue
                            # Cap reached: park every candidate as a
                            # rejected box a human can confirm or requeue
                            # by reason.
                            metrics['combined_no_verdict_cap_hits'] += 1
                            logger.warning(
                                'region_worker_no_verdict_cap',
                                stage='combined',
                                crop_id=t.crop_id,
                                request_id=t.request_id,
                                attempts=no_verdict_cap,
                            )
                            boxes, status, _extra = verdicts_to_boxes(
                                t.candidates,
                                reply.region_boxes if reply is not None else no_verdicts,
                                seq=t.region_box_seq,
                                item_bbox_norm=t.item_bbox_norm,
                                force_resolve=True,
                            )
                            class_update = (
                                None
                                if reply is None
                                else _combined_class_update(
                                    reply, effective_class_names, name_to_id=name_to_id
                                )
                            )
                            # Same trace-tagging as the ordinary verdict
                            # path below -- the cap-reached write is still
                            # a combined-stage reject/accept and must show
                            # up in the detector chain the same way.
                            if any(b.rejection_reason == REJECT_REASON_VERIFIER for b in boxes):
                                metrics['combined_bbox_wrong'] += 1
                            if any(b.state == 'accepted' for b in boxes):
                                t.detection_trace.append(f'{actor}:combined_verify_ok')
                            elif len(boxes) == 1 and boxes[0].rejection_reason:
                                t.detection_trace.append(
                                    f'{actor}:combined_verify_reject:{boxes[0].rejection_reason}'
                                )
                            else:
                                t.detection_trace.append(f'{actor}:combined_verify_reject')
                            # force_resolve=True above always resolves a
                            # real status (never the early-return
                            # no-verdict sentinel) -- narrows the type for
                            # mypy.
                            assert status is not None
                            t.update_doc = _box_list_doc(t, boxes, status, extra=class_update)
                            await out_q.put(t)
                            continue
                        combined_no_verdict.clear(t.crop_id)

                        if any(b.rejection_reason == REJECT_REASON_VERIFIER for b in boxes):
                            metrics['combined_bbox_wrong'] += 1
                        if any(b.state == 'accepted' for b in boxes):
                            t.detection_trace.append(f'{actor}:combined_verify_ok')
                        elif len(boxes) == 1 and boxes[0].rejection_reason:
                            t.detection_trace.append(
                                f'{actor}:combined_verify_reject:{boxes[0].rejection_reason}'
                            )
                        else:
                            t.detection_trace.append(f'{actor}:combined_verify_reject')

                        # Per-box text resolution (VLM + OCR fallback) for
                        # every accepted box, using that box's own
                        # crop-frame bbox. Rejected/no-verdict boxes keep
                        # verdicts_to_boxes' simpler text (the verdict's
                        # own text_reply, or the candidate's OCR hint).
                        resolved_boxes: list[RegionBox] = []
                        for box, cand in zip(boxes, t.candidates, strict=True):
                            if box.state != 'accepted':
                                resolved_boxes.append(box)
                                continue
                            text_doc: dict[str, Any] = {}
                            await apply_region_text(
                                text_doc,
                                ocr=rt.ocr_recognizer,
                                crop_jpeg=t.crop_jpeg,
                                region_in_crop=cand.bbox_in_crop,
                                profile=rt.profile,
                                crop_id=t.crop_id,
                                vlm_text=box.text,
                                vlm_confidence=box.confidence,
                                vlm_available=True,
                                rules=rt.text_rules,
                            )
                            # Text-hint OCR fallback (text_reader='vlm'
                            # only -- other modes already read the
                            # region): the VLM read nothing but the
                            # item-crop OCR hit that seeded this box did.
                            # Forward it so the box stays searchable.
                            apply_text_hint_fallback(
                                text_doc,
                                text=cand.hint_text,
                                confidence=cand.hint_text_confidence,
                                profile=rt.profile,
                                rules=rt.text_rules,
                            )
                            resolved_boxes.append(_box_with_resolved_text(box, text_doc, F))

                        class_update = (
                            None
                            if reply is None
                            else _combined_class_update(
                                reply, effective_class_names, name_to_id=name_to_id
                            )
                        )
                        # We're past the ``extra.get('no_verdict')`` branch
                        # above (which always ``continue``s) -- verdicts_to_boxes
                        # only returns a None status alongside that sentinel,
                        # so a real status is guaranteed here too.
                        assert status is not None
                        t.update_doc = _box_list_doc(t, resolved_boxes, status, extra=class_update)
                        # Preserve the leg's source marker for downstream
                        # consumers via RegionFields.source.
                        if t.candidates:
                            t.update_doc[F.source] = t.candidates[0].source
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
                for _ in flushing:
                    out_q.task_done()
                last_flush = time.monotonic()
                return
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
                cache_hits=state._cache_hits,
                cache_misses=state._cache_misses,
                cache_hit_rate=round(hit_rate, 3),
                vlm_visible_kept=metrics['vlm_visible_kept'],
                vlm_visible_skipped=metrics['vlm_visible_skipped'],
                vlm_visible_skip_rate=round(vis_skip_rate, 3),
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

    try:
        prod_task = asyncio.create_task(producer())
        # Stage A.primary pool: primary-detector + pending_verify
        # routing only. Sized to keep the primary-detector Triton
        # instances saturated; secondary-segmenter work has moved to
        # its own pool below.
        stage_a_tasks = [asyncio.create_task(stage_a_consumer(i)) for i in range(args.concurrency)]
        # Stage A.vlm_visible pool: batched yes/no region-visibility
        # filter for primary-miss / secondary-shape crops. Sized
        # smaller than the combined pool because the visibility prompt
        # is cheap.
        stage_a_visible_tasks = [
            asyncio.create_task(stage_a_vlm_visible(i)) for i in range(vlm_visible_concurrency)
        ]
        # Stage A.secondary pool: secondary-segmenter run on every crop
        # the visibility filter said could contain a region. Sized to
        # keep the secondary-segmenter deployment saturated.
        stage_a_sam_tasks = [
            asyncio.create_task(stage_a_sam_consumer(i)) for i in range(args.concurrency)
        ]
        # Stage B pool: batched combined VLM call (COMBINED_CHUNK
        # crops per upstream call). vlm_concurrency x COMBINED_CHUNK
        # in-flight (16 x 6 = 96 by default). Each crop gets at most
        # ONE combined VLM round-trip here (plus at most ONE yes/no
        # call up in the visibility stage = ≤ 2 per crop total).
        stage_b_tasks = [asyncio.create_task(stage_b_combined(i)) for i in range(vlm_concurrency)]
        writer_task = asyncio.create_task(writer())
        metrics_task = asyncio.create_task(metrics_reporter())
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

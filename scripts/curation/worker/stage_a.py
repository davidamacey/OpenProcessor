"""Stage A.primary (detector routing) and A.vlm_visible (visibility filter)."""

from __future__ import annotations

import asyncio
import time

import structlog

from scripts.curation.worker.pipeline import (
    VISIBLE_CHUNK,
    VLM_CHUNK_DRAIN_TIMEOUT,
    PipelineContext,
    _box_list_doc,
    _drain_chunk,
    _select_candidates,
    _sync_singular_candidate,
)
from scripts.curation.worker.region_text_stage import (
    accept_without_vlm,
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
    bind_task_project,
    unreadable_crop_update,
)
from scripts.curation.worker.verify import task_box_from_stored
from src.config import get_region_fields
from src.config.region_source import CANDIDATE_DETECTOR
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.metrics import (
    OP_STAGE_A_VLM_VISIBLE_DURATION_SECONDS,
    OP_STAGE_REGION_DETECTOR_DURATION_SECONDS,
)
from src.services.labeling.vlm_models import RegionCrop


logger = get_logger('curation_worker')


async def stage_a_consumer(ctx: PipelineContext, consumer_id: int) -> None:
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
    in_q = ctx.in_q
    vlm_visible_q = ctx.vlm_visible_q
    sam_q = ctx.sam_q
    combined_q = ctx.combined_q
    out_q = ctx.out_q
    in_flight = ctx.in_flight
    in_flight_lock = ctx.in_flight_lock
    _rt_for = ctx.rt_for
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
                t.item_ocr_lines = await read_item_lines(rt.ocr_recognizer, t.crop_jpeg, t.crop_id)
                t.item_text_update = item_text_fields(
                    t.item_ocr_lines, min_confidence=rt.item_text_min_conf
                )

            is_secondary = _is_secondary_shape(t)

            # Path 1: pending_verify — already has a candidate awaiting
            # VLM verification; straight to the combined call (no
            # fresh detection needed). W8 B1 fix: the candidate comes
            # from this item's STORED ``region_boxes`` -- every box
            # whose state is ``proposed`` (the real source of truth,
            # incl. a human's box set via `PUT /crops/{id}/regions`)
            # -- keeping its box_id.
            proposed_stored = [b for b in t.stored_boxes if b.state == 'proposed']
            if t.region_status in _PENDING_VERIFICATION_ALIASES and proposed_stored:
                t.candidates = [
                    task_box_from_stored(b, item_bbox_norm=t.item_bbox_norm)
                    for b in proposed_stored
                ]
                # B1: this pass only re-verifies the stored
                # `proposed` box(es) -- any sibling (already
                # accepted/rejected, or a second proposed box not
                # selected here) must be merged back at write
                # time, never silently replaced.
                t.pending_merge = True
                # W8c B1 fix (2026-09-28 re-review): this is the
                # ONE place that is a genuine Path-1 re-verify --
                # distinct from `pending_merge` alone, which every
                # fresh-detection pass below also sets (for its
                # own, different, keep-human/replace-machine
                # reason). `runner.py`'s not-visible branch reads
                # this to resolve the candidate as `rejected`
                # (keeping its stored id) instead of writing an
                # empty box list.
                t.reverify = True
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

            # W8c (r1 fix, corrected 2026-09-28 re-review -- M2): every
            # fresh-detection path below (Path 2 and Path 3, plus the
            # text-hint re-pass they can fall into) keeps this item's
            # human-owned stored boxes and replaces its machine-sourced
            # ones with this pass's own findings (M1 fix --
            # `bulk_writer._merge` branches on `pending_merge` +
            # `reverify`, see `_ItemTask`'s docstrings). Requeuing a
            # terminal item (e.g. `verify_rejected` -> `pending_detection`)
            # intentionally leaves a human-sourced box in `region_boxes`
            # (`region_requeue.apply_requeue`); without this, this
            # pass's own candidates would silently discard it. A no-op
            # for the common case (no stored boxes at all).
            #
            # CORRECTION (M2): reaching this line does NOT prove
            # `region_status` is a `pending_detection` alias -- Path 1's
            # `if` above only `continue`s when it found something to
            # re-verify (`proposed_stored`); a `pending_verification`
            # alias item with neither (which should no longer occur --
            # `region_requeue.apply_requeue` now re-proposes a box
            # before ever setting that target status, or skips the item
            # entirely -- but stale/pre-fix data or a direct write can
            # still produce one) falls through to here too and runs a
            # fresh detection instead of a re-verify. Log it so that
            # silent mismatch is at least visible.
            if t.region_status in _PENDING_VERIFICATION_ALIASES:
                logger.warning(
                    'region_pending_verification_fallthrough',
                    crop_id=t.crop_id,
                    detail='pending_verification item had no proposed box to '
                    're-verify; running a fresh detection pass instead (M2)',
                )
            t.pending_merge = True

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


async def stage_a_vlm_visible(ctx: PipelineContext, consumer_id: int) -> None:
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
    vlm_visible_q = ctx.vlm_visible_q
    sam_q = ctx.sam_q
    out_q = ctx.out_q
    in_flight = ctx.in_flight
    in_flight_lock = ctx.in_flight_lock
    metrics = ctx.metrics
    no_verdict_cap = ctx.no_verdict_cap
    visible_no_verdict = ctx.visible_no_verdict
    _rt_for = ctx.rt_for

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
                            _t.mark_vlm_called(rt.vlm_identity)
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

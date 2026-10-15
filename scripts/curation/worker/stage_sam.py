"""Stage A.secondary: segmenter run plus the text-hint OCR fallback."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from scripts.curation.worker.cascade import _resegment_from_text_hint
from scripts.curation.worker.client import SegmenterUnavailable
from scripts.curation.worker.fairness import is_region_stage_paused
from scripts.curation.worker.pipeline import (
    PipelineContext,
    _box_list_doc,
    _select_candidates,
    _sync_singular_candidate,
)
from scripts.curation.worker.region_text_stage import _box_with_resolved_text, apply_region_text
from scripts.curation.worker.state import _ItemTask, bind_task_project, unreadable_crop_update
from scripts.curation.worker.verify import (
    _SKIP_VLM_VERIFY_SECONDARY_SCORE,
    TaskBoxInput,
    _bbox_shape_is_plausible,
    item_verification_fields,
)
from scripts.curation.worker.verify_choice import dispatch_verification, effective_verify
from src.config import get_region_fields
from src.config.region_source import CANDIDATE_SEGMENTER, CANDIDATE_SEGMENTER_TEXT_HINT
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.domain.region_boxes import RegionBox, new_box_placeholder
from src.services.curation.metrics import OP_STAGE_A_SEGMENTER_DURATION_SECONDS
from src.services.curation.ops_metrics import record_segmenter_request
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm
from src.services.detection.segmenter_gate import RUN


logger = get_logger('curation_worker')


async def stage_a_sam_consumer(ctx: PipelineContext, consumer_id: int) -> None:
    """Stage A.secondary: secondary-segmenter run + text-hint OCR fallback.

    Runs on every crop the visibility filter said could contain a
    region of interest (primary-miss / secondary-shape crops that
    came through ``stage_a_vlm_visible``). High-confidence
    segmenter hits with a region-shaped bbox short-circuit the VLM
    entirely (zero combined calls — same policy as the pre-collapse
    pipeline; see ``_SKIP_VLM_VERIFY_SECONDARY_SCORE``). Everything
    else routes to Stage B.combined for a single VLM round-trip.
    """
    while True:
        t = await ctx.sam_q.get()
        if t is None:
            ctx.sam_q.task_done()
            return
        bind_task_project(t)
        structlog.contextvars.bind_contextvars(request_id=t.request_id)
        try:
            await _route_item(ctx, consumer_id, t)
        except Exception as exc:
            logger.warning(
                'stage_a_sam_failed',
                consumer_id=consumer_id,
                crop_id=t.crop_id,
                error=str(exc),
            )
            async with ctx.in_flight_lock:
                ctx.in_flight.discard(t.crop_id)
            ctx.sam_q.task_done()
        finally:
            structlog.contextvars.unbind_contextvars('request_id')


async def _release_unprocessed(ctx: PipelineContext, t: _ItemTask) -> None:
    # Left exactly as fetched (``pending_detection``): the producer
    # picks it up again once the stage runs.
    async with ctx.in_flight_lock:
        ctx.in_flight.discard(t.crop_id)
    ctx.sam_q.task_done()


async def _leave_pending_segmenter_down(
    ctx: PipelineContext, consumer_id: int, t: _ItemTask, exc: SegmenterUnavailable
) -> None:
    # Infrastructure failure (a request failed or every secondary-
    # segmenter host is UNHEALTHY). Do NOT mark the crop terminal --
    # leave region_status unchanged so it stays in pending_detection
    # for the next cascade pass once a host recovers. Drop from
    # in_flight + sleep so the producer can re-fetch and we don't
    # spin a hot loop while every host is down.
    logger.error(
        'stage_a_sam_all_hosts_down',
        consumer_id=consumer_id,
        crop_id=t.crop_id,
        error=str(exc),
    )
    async with ctx.in_flight_lock:
        ctx.in_flight.discard(t.crop_id)
    ctx.sam_q.task_done()
    await asyncio.sleep(1.0)


async def _route_item(ctx: PipelineContext, consumer_id: int, t: _ItemTask) -> None:
    """One task through the segmenter pass. Every path ends with exactly one
    ``sam_q.task_done()``, whichever queue (or release) the task leaves by."""
    F = get_region_fields()
    rt = ctx.rt_for(t)
    if t.crop_jpeg is None:
        t.update_doc = unreadable_crop_update(t)
        await ctx.out_q.put(t)
        ctx.sam_q.task_done()
        return

    if t.project is not None and is_region_stage_paused(t.project):
        await _release_unprocessed(ctx, t)
        return
    gate_verdict = await ctx.crop_gate.decide(t, rt.profile) if rt.segmenter.enabled else RUN
    if not gate_verdict.run:
        t.detection_trace.append(f'gate:{gate_verdict.label}')
        t.update_doc = _box_list_doc(
            t,
            [],
            RegionStatus.NO_REGION_BOX,
            extra={F.gate_skip: gate_verdict.label},
        )
        await ctx.out_q.put(t)
        ctx.sam_q.task_done()
        return

    segmented = await _segment(ctx, consumer_id, rt, t)
    if segmented is None:
        return
    sam_selected, sam_elapsed = segmented
    if rt.segmenter.enabled:
        ctx.crop_gate.observe(t, rt.profile, hit=bool(sam_selected), seconds=sam_elapsed)
    if sam_selected:
        await _route_hits(ctx, rt, t, sam_selected)
        return

    if not rt.text_hint_on:
        # No text-hint re-pass: the segmenter's miss is final.
        if rt.segmenter.enabled:
            t.detection_trace.append(f'{rt.profile.segmenter_name}:miss')
    elif await _text_hint_pass(ctx, consumer_id, rt, t, t.crop_jpeg):
        return

    # Nothing found by any detector → no_region_box.
    t.update_doc = _box_list_doc(t, [], RegionStatus.NO_REGION_BOX)
    await ctx.out_q.put(t)
    ctx.sam_q.task_done()


async def _segment(
    ctx: PipelineContext, consumer_id: int, rt: Any, t: _ItemTask
) -> tuple[list[TaskBoxInput], float] | None:
    """Run the secondary segmenter on the crop and select its candidates.
    ``None`` when every host is down (the task was left pending)."""
    _sam_t0 = time.monotonic()
    try:
        raw_sam_cands = await rt.segmenter.segment_multi(t.crop_jpeg)
    except SegmenterUnavailable as exc:
        OP_STAGE_A_SEGMENTER_DURATION_SECONDS.labels(outcome='error').observe(
            time.monotonic() - _sam_t0
        )
        record_segmenter_request('error', time.monotonic() - _sam_t0)
        await _leave_pending_segmenter_down(ctx, consumer_id, t, exc)
        return None
    except Exception:
        OP_STAGE_A_SEGMENTER_DURATION_SECONDS.labels(outcome='error').observe(
            time.monotonic() - _sam_t0
        )
        record_segmenter_request('error', time.monotonic() - _sam_t0)
        raise
    _sam_elapsed = time.monotonic() - _sam_t0
    OP_STAGE_A_SEGMENTER_DURATION_SECONDS.labels(
        outcome='hit' if raw_sam_cands else 'miss'
    ).observe(_sam_elapsed)
    record_segmenter_request('hit' if raw_sam_cands else 'miss', _sam_elapsed)
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
    return sam_selected, _sam_elapsed


async def _queue_for_verification(ctx: PipelineContext, rt: Any, t: _ItemTask) -> None:
    """Candidates set on ``t``: one combined VLM call, or straight to the writer
    (see :mod:`~scripts.curation.worker.verify_choice`)."""
    await dispatch_verification(ctx, rt, t)


async def _route_hits(
    ctx: PipelineContext, rt: Any, t: _ItemTask, sam_selected: list[TaskBoxInput]
) -> None:
    F = get_region_fields()
    # High-conf-skip: bypass VLM verify entirely, but only
    # when there is exactly ONE candidate to auto-accept
    # -- with N>1 candidates the VLM is the adjudicator
    # (that's the point of offering more than one), so the
    # skip shortcut never applies to a multi-candidate set.
    top = sam_selected[0]
    if (
        len(sam_selected) == 1
        and effective_verify(t) == 'vlm'
        and top.score >= _SKIP_VLM_VERIFY_SECONDARY_SCORE
        and _bbox_shape_is_plausible(top.bbox_in_crop)
    ):
        t.detection_trace.append(f'{rt.profile.segmenter_name}:hit')
        t.detection_trace.append(f'{rt.profile.segmenter_name}:skip_vlm_verify')
        box = RegionBox(
            box_id=new_box_placeholder(0),
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
            vlm_model=rt.vlm_model,
            rules=rt.text_rules,
        )
        box = _box_with_resolved_text(box, text_doc)
        # skip-verify never calls the VLM, so
        # verified/auto_confirmed stay False.
        t.update_doc = _box_list_doc(
            t,
            [box],
            RegionStatus.DETECTED,
            extra={
                F.skip_verify: True,
                **item_verification_fields(verified=False, verifier=None),
            },
        )
        await ctx.out_q.put(t)
        ctx.sam_q.task_done()
        return
    # Else: queue the secondary-segmenter candidate(s) for
    # a combined VLM call.
    t.detection_trace.append(f'{rt.profile.segmenter_name}:hit')
    t.candidates = sam_selected
    _sync_singular_candidate(t)
    await _queue_for_verification(ctx, rt, t)
    ctx.sam_q.task_done()


async def _text_hint_pass(
    ctx: PipelineContext, consumer_id: int, rt: Any, t: _ItemTask, crop_jpeg: bytes
) -> bool:
    """Secondary segmenter missed globally; re-prompt it
    with a tight sub-crop around an OCR text hint. The
    OCR-detection bbox is no longer trusted; the
    segmenter produces the final geometry. OCR text
    rides along for storage.

    True when the task left the stage here (routed with the hinted candidate,
    or left pending because the segmenter is down); False to fall through to
    ``no_region_box``."""
    if t.item_ocr_lines is not None:
        ocr_regions = rt.ocr_recognizer.regions_from_lines(t.item_ocr_lines)
    else:
        try:
            ocr_regions = await rt.ocr_recognizer.detect_regions(crop_jpeg)
        except Exception as exc:
            logger.warning('text_hint_ocr_failed', crop_id=t.crop_id, error=str(exc))
            ocr_regions = []
    ocr_pick = rt.ocr_recognizer.pick_best_text_region(ocr_regions) if ocr_regions else None
    if ocr_pick is None:
        if ocr_regions:
            t.detection_trace.append(f'{rt.profile.ocr_rec_model}:text_hint:no_region_shape')
        else:
            t.detection_trace.append(f'{rt.profile.ocr_rec_model}:text_hint:miss')
        return False
    t.detection_trace.append(f'{rt.profile.ocr_rec_model}:text_hint:hit')
    try:
        sub_cand, _sub_box = await _resegment_from_text_hint(
            crop_jpeg, ocr_pick.bbox_norm, rt.segmenter
        )
    except SegmenterUnavailable as exc:
        await _leave_pending_segmenter_down(ctx, consumer_id, t, exc)
        return True
    if sub_cand is None:
        t.detection_trace.append(f'{rt.profile.segmenter_name}:text_hint:miss')
        return False
    t.candidates = [
        TaskBoxInput(
            bbox_in_crop=sub_cand.bbox_norm,
            bbox_in_source=crop_norm_to_source_norm(sub_cand.bbox_norm, t.item_bbox_norm),
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
    await _queue_for_verification(ctx, rt, t)
    ctx.sam_q.task_done()
    return True

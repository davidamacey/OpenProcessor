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
from scripts.curation.worker.region_text_stage import (
    _box_with_resolved_text,
    accept_without_vlm,
    apply_region_text,
)
from scripts.curation.worker.state import _ItemTask, bind_task_project, unreadable_crop_update
from scripts.curation.worker.verify import (
    _SKIP_VLM_VERIFY_SECONDARY_SCORE,
    TaskBoxInput,
    _bbox_shape_is_plausible,
    item_verification_fields,
)
from src.config import get_region_fields
from src.config.region_source import CANDIDATE_SEGMENTER, CANDIDATE_SEGMENTER_TEXT_HINT
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.metrics import OP_STAGE_A_SEGMENTER_DURATION_SECONDS
from src.services.curation.ops_metrics import record_segmenter_request
from src.services.curation.region_boxes import RegionBox, new_box_placeholder
from src.services.detection.cascade_detect import crop_norm_to_source_norm
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
    sam_q = ctx.sam_q
    combined_q = ctx.combined_q
    out_q = ctx.out_q
    in_flight = ctx.in_flight
    in_flight_lock = ctx.in_flight_lock
    crop_gate = ctx.crop_gate
    _rt_for = ctx.rt_for

    F = get_region_fields()

    async def release_unprocessed(t: _ItemTask) -> None:
        # Left exactly as fetched (``pending_detection``): the producer
        # picks it up again once the stage runs.
        async with in_flight_lock:
            in_flight.discard(t.crop_id)
        sam_q.task_done()

    async def leave_pending_segmenter_down(t: _ItemTask, exc: SegmenterUnavailable) -> None:
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
        async with in_flight_lock:
            in_flight.discard(t.crop_id)
        sam_q.task_done()
        await asyncio.sleep(1.0)

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

            if t.project is not None and is_region_stage_paused(t.project):
                await release_unprocessed(t)
                continue
            gate_verdict = await crop_gate.decide(t, rt.profile) if rt.segmenter.enabled else RUN
            if not gate_verdict.run:
                t.detection_trace.append(f'gate:{gate_verdict.label}')
                t.update_doc = _box_list_doc(
                    t,
                    [],
                    RegionStatus.NO_REGION_BOX,
                    extra={F.gate_skip: gate_verdict.label},
                )
                await out_q.put(t)
                sam_q.task_done()
                continue

            _sam_t0 = time.monotonic()
            try:
                raw_sam_cands = await rt.segmenter.segment_multi(t.crop_jpeg)
            except SegmenterUnavailable as exc:
                OP_STAGE_A_SEGMENTER_DURATION_SECONDS.labels(outcome='error').observe(
                    time.monotonic() - _sam_t0
                )
                record_segmenter_request('error', time.monotonic() - _sam_t0)
                await leave_pending_segmenter_down(t, exc)
                continue
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
            if rt.segmenter.enabled:
                crop_gate.observe(t, rt.profile, hit=bool(sam_selected), seconds=_sam_elapsed)
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
                    # W8 M2: skip-verify never calls the VLM --
                    # verified/auto_confirmed stay False, same as the
                    # pre-W8 skip-verify write.
                    t.update_doc = _box_list_doc(
                        t,
                        [box],
                        RegionStatus.DETECTED,
                        extra={
                            F.skip_verify: True,
                            **item_verification_fields(verified=False, verifier=None),
                        },
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
                        logger.warning('text_hint_ocr_failed', crop_id=t.crop_id, error=str(exc))
                        ocr_regions = []
                ocr_pick = (
                    rt.ocr_recognizer.pick_best_text_region(ocr_regions) if ocr_regions else None
                )
                if ocr_pick is not None:
                    t.detection_trace.append(f'{rt.profile.ocr_rec_model}:text_hint:hit')
                    try:
                        sub_cand, _sub_box = await _resegment_from_text_hint(
                            t.crop_jpeg, ocr_pick.bbox_norm, rt.segmenter
                        )
                    except SegmenterUnavailable as exc:
                        await leave_pending_segmenter_down(t, exc)
                        continue
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

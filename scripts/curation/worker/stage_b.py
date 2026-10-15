"""Stage B: the batched combined VLM call."""

from __future__ import annotations

import time
import uuid
from functools import partial
from typing import Any

import structlog

from scripts.curation.worker.combined_resolve import resolve_combined_reply
from scripts.curation.worker.pipeline import (
    COMBINED_CHUNK,
    VLM_CHUNK_DRAIN_TIMEOUT,
    PipelineContext,
    _box_list_doc,
    _drain_chunk,
    _should_classify,
)
from scripts.curation.worker.state import (
    RegionProfileNotConfiguredError,
    _ItemTask,
    bind_task_project,
    bound_class_catalog,
    unreadable_crop_update,
)
from scripts.curation.worker.verify import candidate_actor, chain_entry
from src.core.logging import get_logger
from src.services.curation.assist.slot_write import RunMeta
from src.services.curation.metrics import OP_STAGE_B_VLM_VERIFY_DURATION_SECONDS
from src.services.labeling.vlm_models import CombinedCrop
from src.utils.timeutil import now_iso


logger = get_logger('curation_worker')


async def stage_b_combined(ctx: PipelineContext, consumer_id: int) -> None:
    """Stage B: batched combined VLM call (class + region-verify + OCR).

    Replaces the legacy vlm_visible + vlm_verify two-call
    cascade with ONE VLM round-trip per crop. Every candidate box
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
        follows this precedence (accepted > false_positive >
        proposed > rejected > empty). ``combined_bbox_wrong`` counts
        crops where at least one candidate's box was visible-
        elsewhere-rejected.
      - the call itself failed (transport) -> every crop in the chunk
        is retried, never counted toward the cap.
    """
    carry: list[_ItemTask] = []
    while True:
        chunk, poisoned = await _drain_chunk(
            ctx.combined_q,
            chunk_size=COMBINED_CHUNK,
            drain_timeout=VLM_CHUNK_DRAIN_TIMEOUT,
            carry=carry,
        )
        if chunk:
            await _process_chunk(ctx, consumer_id, chunk)
        if poisoned:
            return


async def _release(ctx: PipelineContext, chunk: list[_ItemTask]) -> None:
    async with ctx.in_flight_lock:
        for t in chunk:
            ctx.in_flight.discard(t.crop_id)


async def _process_chunk(ctx: PipelineContext, consumer_id: int, chunk: list[_ItemTask]) -> None:
    """One single-project chunk: the VLM call, then each task's resolution."""
    # Single-project chunk (see _drain_chunk): classify
    # against that project's own registry.
    bind_task_project(chunk[0])
    try:
        rt = ctx.rt_for(chunk[0])
    except RegionProfileNotConfiguredError as exc:
        logger.warning(
            'stage_b_combined_no_runtime',
            project=chunk[0].project.slug,
            error=str(exc),
        )
        await _release(ctx, chunk)
        return
    class_names, name_to_id = bound_class_catalog()
    registry_loaded = bool(class_names)

    combined_crops, bad_indices = _combined_crops(chunk, registry_loaded=registry_loaded)
    replies_by_id: dict[str, Any] = {}
    if combined_crops:
        replies = await _label_combined(ctx, rt, consumer_id, chunk, combined_crops, class_names)
        if replies is None:
            return
        replies_by_id = replies

    for i, t in enumerate(chunk):
        structlog.contextvars.bind_contextvars(request_id=t.request_id)
        try:
            if i in bad_indices:
                # Defensive — Stage A should always set these.
                t.update_doc = unreadable_crop_update(t)
                await ctx.out_q.put(t)
                continue
            await _resolve_task(
                ctx,
                rt,
                t,
                reply=replies_by_id.get(t.crop_id),
                class_names=class_names,
                name_to_id=name_to_id,
                registry_loaded=registry_loaded,
            )
        finally:
            structlog.contextvars.unbind_contextvars('request_id')


def _combined_crops(
    chunk: list[_ItemTask], *, registry_loaded: bool
) -> tuple[list[CombinedCrop], list[int]]:
    """Build CombinedCrop payloads (parent item JPEG +
    candidate bbox for overlay drawing). Per-crop
    ``classify`` flag tells the VLM whether to fill
    class_id (low-conf primary / unknown source) or skip
    it (high-conf primary crops where the caller already
    trusts the class). Also the indices of tasks without a crop."""
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
    return combined_crops, bad_indices


async def _label_combined(
    ctx: PipelineContext,
    rt: Any,
    consumer_id: int,
    chunk: list[_ItemTask],
    combined_crops: list[CombinedCrop],
    class_names: list[str],
) -> dict[str, Any] | None:
    """The batched VLM call. ``None`` after a transport failure, with the
    whole chunk released so the producer re-fetches it."""
    batch_request_ids = [t.request_id for t in chunk]
    _vlm_t0 = time.monotonic()
    try:
        if rt.vlm is None:
            msg = 'combined stage fed without a VLM'
            raise RuntimeError(msg)
        replies_by_id: dict[str, Any] = await rt.vlm.label_combined_batch(
            combined_crops,
            class_names=class_names or None,
        )
        # only a write this call actually
        # informed gets stamped `vlm_prompt_pack` downstream.
        for _t in chunk:
            if _t.crop_jpeg is not None:
                _t.mark_vlm_called(rt.vlm_identity)
        _vlm_elapsed = time.monotonic() - _vlm_t0
        OP_STAGE_B_VLM_VERIFY_DURATION_SECONDS.labels(outcome='ok').observe(_vlm_elapsed)
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
        await _release(ctx, chunk)
        return None
    return replies_by_id


def combined_slot_meta(rt: Any) -> RunMeta | None:
    """Who a combined call's `assist` slot is attributed to: the runtime's endpoint and pack."""
    identity = rt.vlm_identity
    if identity is None:
        return None
    name, revision = rt.pack_ref
    return RunMeta(
        run_id=f'rw_{uuid.uuid4().hex[:12]}',
        created_at=now_iso(),
        vlm_endpoint=identity.endpoint_ref,
        vlm_model=identity.model,
        prompt_pack=name,
        prompt_pack_revision=revision,
    )


async def _resolve_task(
    ctx: PipelineContext,
    rt: Any,
    t: _ItemTask,
    *,
    reply: Any,
    class_names: list[str],
    name_to_id: dict[str, int],
    registry_loaded: bool,
) -> None:
    """Resolve one task's reply into its update doc and queue it for the writer
    (or release it for a retry when no candidate got a verdict yet)."""
    # Decide per-crop class-field side: only honor
    # the VLM's class when we asked it to classify.
    effective_class_names = (
        class_names if _should_classify(t, registry_loaded=registry_loaded) else None
    )

    # Canonical detector name + version for provenance
    # (the whole item's candidates share one leg per
    # pass -- Path 1/2/segmenter never mix in the same
    # combined_q push).
    actor = candidate_actor(t.candidates[0]) if t.candidates else 'unknown'
    # The candidate's detector gets exactly one ``:hit``
    # (Stage A records it for fresh detections; an
    # ingest-time box awaiting verification has none).
    for entry in chain_entry(actor, 'hit'):
        if entry not in t.detection_trace:
            t.detection_trace.append(entry)

    resolve = partial(
        resolve_combined_reply,
        t.candidates,
        reply,
        item_bbox_norm=t.item_bbox_norm,
        reverify=t.reverify,
        effective_class_names=effective_class_names,
        name_to_id=name_to_id,
        vlm_model=rt.vlm_model,
        slot_meta=combined_slot_meta(rt),
        profile=rt.profile,
        rules=rt.text_rules,
        ocr=rt.ocr_recognizer,
        crop_jpeg=t.crop_jpeg,
        crop_id=t.crop_id,
    )
    resolution = await resolve()
    if resolution.outcome == 'no_verdict':
        # No candidate got any verdict at all -- the
        # entry is missing/unparseable, or the VLM saw
        # a region but answered null/nothing on every
        # box. Not a reject: leave the item pending --
        # drop it from in_flight so the next producer
        # poll retries it -- until the no-verdict cap.
        if reply is None:
            ctx.metrics['combined_parse_failure'] += 1
            logger.warning(
                'stage_b_combined_parse_failure',
                crop_id=t.crop_id,
                request_id=t.request_id,
            )
        else:
            ctx.metrics['combined_no_bbox_verdict'] += 1
            logger.info(
                'stage_b_combined_no_bbox_verdict',
                crop_id=t.crop_id,
                request_id=t.request_id,
            )
        if not ctx.combined_no_verdict.record(t.crop_id):
            async with ctx.in_flight_lock:
                ctx.in_flight.discard(t.crop_id)
            return
        # Cap reached: park every candidate as a
        # rejected box a human can confirm or requeue
        # by reason.
        ctx.metrics['combined_no_verdict_cap_hits'] += 1
        logger.warning(
            'region_worker_no_verdict_cap',
            stage='combined',
            crop_id=t.crop_id,
            request_id=t.request_id,
            attempts=ctx.no_verdict_cap,
        )
        resolution = await resolve(force_resolve=True)
    else:
        ctx.combined_no_verdict.clear(t.crop_id)
    if resolution.no_region_visible:
        ctx.metrics['combined_no_region_visible'] += 1
    if resolution.bbox_wrong:
        ctx.metrics['combined_bbox_wrong'] += 1
    t.detection_trace.extend(resolution.trace)
    assert resolution.status is not None
    t.update_doc = _box_list_doc(t, resolution.boxes, resolution.status, extra=resolution.extra)
    await ctx.out_q.put(t)

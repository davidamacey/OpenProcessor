"""Run a region profile's legs over one stored crop and preview the item
(W5, ``POST /region_profiles/test``).

The legs run as the worker runs them: the detector leg when the profile
names a detector model, the segmenter leg when there is a prompt and the
detector produced nothing (the worker does not run the segmenter after a
detector hit), each followed by the worker's own candidate selection. The
write the worker would make for the selected boxes is built from the same
functions the worker calls (``verdicts_to_boxes`` / ``resolve_combined_reply``
-> ``box_pass_update`` -> ``finalize_region_write``), then laid over the
stored item. Nothing is written.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config import get_region_fields
from src.config.region_source import CANDIDATE_DETECTOR, CANDIDATE_SEGMENTER
from src.config.region_state import RegionStatus
from src.services.curation.class_write_guard import class_state_token
from src.services.curation.region_box_pass import (
    box_pass_update,
    finalize_region_write,
    worker_stamps,
)
from src.services.curation.region_boxes import read_boxes
from src.services.curation.region_preview import (
    LegCandidate,
    LegSelection,
    candidate_wires,
    preview_item,
    select_leg_candidates,
)
from src.services.detection.cascade_detect import RegionDetector, crop_norm_to_source_norm
from src.services.detection.segmenter_http import (
    DEFAULT_MAX_CANDIDATES,
    SegmenterCallError,
    segment_once,
)
from src.services.labeling.region_overlay import VlmBoxVerdict


if TYPE_CHECKING:
    from collections.abc import Callable

    from scripts.curation.worker.verify import TaskBoxInput
    from src.config import DetectionProfile
    from src.services.labeling.vlm_labeler import VlmLabeler
    from src.services.labeling.vlm_probe import ProbeResult


@dataclass
class Leg:
    leg: str
    status: str
    reason: str | None = None
    elapsed_ms: float | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selection: LegSelection | None = None
    detector: str = ''
    detector_version: str = ''
    box_source: str = ''
    #: The raw leg produced something (the worker's ``<leg>:hit`` / ``:miss``).
    hit: bool = False


@dataclass(frozen=True)
class VerifyContext:
    """What the optional combined VLM call needs."""

    labeler: VlmLabeler
    pack_stamp: str
    class_names: list[str]
    name_to_id: dict[str, int]
    vlm_name: str | None = None


@dataclass
class RegionRun:
    legs: list[Leg]
    preview: dict[str, Any]
    basis: str
    probe: ProbeResult | None = None


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


async def _detector_leg(
    source: dict[str, Any], profile: DetectionProfile, jpeg: bytes, pool: Callable[[], Any]
) -> Leg:
    leg = Leg(
        leg='detector',
        status='skipped',
        detector=profile.detector_model,
        detector_version=profile.detector_version,
        box_source=CANDIDATE_DETECTOR,
    )
    if not profile.detector_model:
        leg.reason = 'no_detector_model'
        return leg
    started = time.perf_counter()
    try:
        found = await RegionDetector(pool(), profile).detect_multi(jpeg)
    except Exception as exc:
        leg.status, leg.reason, leg.elapsed_ms = (
            'error',
            f'{type(exc).__name__}: {exc}',
            _ms(started),
        )
        return leg
    leg.status, leg.elapsed_ms, leg.hit = 'ok', _ms(started), bool(found)
    raw = [
        LegCandidate(bbox_norm=c.bbox_norm, score=c.score, mask_iou=c.rectangularity) for c in found
    ]
    leg.selection = select_leg_candidates(
        raw, source=CANDIDATE_DETECTOR, profile=profile, min_score=profile_floor(profile)
    )
    leg.candidates = candidate_wires(
        source,
        leg.selection,
        detector=leg.detector,
        detector_version=leg.detector_version,
        box_source=CANDIDATE_DETECTOR,
    )
    return leg


def profile_floor(profile: DetectionProfile) -> float:
    """The detector leg's selection floor (``confidence_floor``)."""
    return profile.confidence_floor


async def _segmenter_leg(
    source: dict[str, Any],
    profile: DetectionProfile,
    jpeg: bytes,
    *,
    prompt: str,
    url: str | None,
    detector_hit: bool,
) -> Leg:
    leg = Leg(
        leg='segmenter',
        status='skipped',
        detector=profile.segmenter_name,
        detector_version=profile.segmenter_version,
        box_source=CANDIDATE_SEGMENTER,
    )
    if detector_hit:
        leg.reason = 'detector_hit'
        return leg
    if not prompt:
        leg.reason = 'no_segmenter_prompt'
        return leg
    if url is None:
        leg.reason = 'segmenter_not_configured'
        return leg
    started = time.perf_counter()
    try:
        found = await segment_once(
            url, jpeg, prompt, max_candidates=DEFAULT_MAX_CANDIDATES, return_masks=True
        )
    except SegmenterCallError as exc:
        leg.status, leg.reason, leg.elapsed_ms = 'error', str(exc), _ms(started)
        return leg
    leg.status, leg.elapsed_ms, leg.hit = 'ok', _ms(started), bool(found)
    raw = [
        LegCandidate(
            bbox_norm=c.bbox_norm,
            score=c.score,
            mask_iou=c.mask_iou,
            mask_polygon=c.mask_polygon,
        )
        for c in found
    ]
    leg.selection = select_leg_candidates(
        raw, source=CANDIDATE_SEGMENTER, profile=profile, min_score=0.0
    )
    leg.candidates = candidate_wires(
        source,
        leg.selection,
        detector=leg.detector,
        detector_version=leg.detector_version,
        box_source=CANDIDATE_SEGMENTER,
    )
    return leg


def _task_boxes(leg: Leg, parent: tuple[float, float, float, float]) -> list[TaskBoxInput]:
    from scripts.curation.worker.verify import TaskBoxInput

    assert leg.selection is not None
    return [
        TaskBoxInput(
            bbox_in_crop=c.bbox_norm,
            bbox_in_source=crop_norm_to_source_norm(c.bbox_norm, parent),
            score=c.score,
            detector=leg.detector,
            detector_version=leg.detector_version,
            source=leg.box_source,
        )
        for c in leg.selection.selection.selected
    ]


def _parent_box(source: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = [float(v) for v in source.get('bbox_norm') or (0.0, 0.0, 1.0, 1.0)]
    return bbox[0], bbox[1], bbox[2], bbox[3]


async def run_region_test(
    *,
    source: dict[str, Any],
    crop_id: str,
    jpeg: bytes,
    profile: DetectionProfile,
    stamp_name: str | None,
    stamp_revision: int | None,
    segmenter_prompt: str,
    segmenter_url: str | None,
    triton_pool: Callable[[], Any],
    verify: VerifyContext | None,
) -> RegionRun:
    """The legs, and the item as the worker would leave it. ``stamp_name`` /
    ``stamp_revision`` are the profile provenance the write would carry
    (``None`` for a draft, which was never saved)."""
    from scripts.curation.worker.combined_resolve import resolve_combined_reply, should_classify
    from scripts.curation.worker.verify import item_verification_fields, verdicts_to_boxes

    detector = await _detector_leg(source, profile, jpeg, triton_pool)
    detector_hit = bool(detector.selection and detector.selection.selection.selected)
    segmenter = await _segmenter_leg(
        source,
        profile,
        jpeg,
        prompt=segmenter_prompt,
        url=segmenter_url,
        detector_hit=detector_hit,
    )
    legs = [detector, segmenter]
    producer = detector if detector_hit else segmenter
    parent = _parent_box(source)
    selected = _task_boxes(producer, parent) if producer.selection is not None else []

    F = get_region_fields()
    trace: list[str] = []
    if detector.status == 'ok':
        trace.append(f'{detector.detector}:{"hit" if detector.hit else "miss"}')
    if segmenter.status == 'ok' and not detector_hit:
        trace.append(f'{segmenter.detector}:{"hit" if segmenter.hit else "miss"}')

    probe: ProbeResult | None = None
    basis = 'selection_accepted'
    extra: dict[str, Any]
    if not selected:
        boxes: list[Any] = []
        status = RegionStatus.NO_REGION_BOX
        extra = {}
    elif verify is None:
        verdicts = [
            VlmBoxVerdict(box=i, bbox_correct=True, confidence=None)
            for i in range(1, len(selected) + 1)
        ]
        boxes, derived, _ = verdicts_to_boxes(selected, verdicts, item_bbox_norm=parent)
        assert derived is not None
        status = derived
        extra = item_verification_fields(verified=False, verifier=None)
        trace.append(f'{selected[0].detector}:accepted_unverified')
    else:
        from src.services.detection.region_text_rules import region_text_rules
        from src.services.labeling.vlm_probe import ProbeCrop, probe as run_probe

        basis = 'vlm_verdicts'
        classify = should_classify(
            class_validated=bool(source.get('class_validated')),
            stored_class_source=str(source.get('class_source') or ''),
            test_holdout=bool(source.get('test_holdout')),
            class_confidence=float(source.get('confidence') or 0.0),
            registry_loaded=bool(verify.class_names),
        )
        names = verify.class_names if classify else None
        probe = await run_probe(
            verify.labeler,
            'combined',
            [
                ProbeCrop(
                    crop_id=crop_id, jpeg=jpeg, region_boxes=[c.bbox_in_crop for c in selected]
                )
            ],
            class_names=names,
        )
        reply = probe.parsed[0].value
        resolution = await resolve_combined_reply(
            selected,
            reply,
            item_bbox_norm=parent,
            reverify=False,
            effective_class_names=names,
            name_to_id=verify.name_to_id,
            vlm_model=verify.labeler.identity.model,
            profile=profile,
            rules=region_text_rules(profile),
            ocr=None,
            crop_jpeg=jpeg,
            crop_id=crop_id,
        )
        if resolution.outcome == 'no_verdict':
            # The worker retries (up to its cap) and writes nothing yet.
            return RegionRun(
                legs=legs, preview=preview_item(source, {}, crop_id), basis=basis, probe=probe
            )
        assert resolution.status is not None
        boxes, status, extra = resolution.boxes, resolution.status, resolution.extra
        trace.extend(resolution.trace)

    box_pass = box_pass_update(
        source,
        boxes,
        reverify=False,
        merge_machine_boxes=True,
        baseline=read_boxes(source, F),
        status=None,
        empty_status=status,
    )
    update = {**extra, **box_pass.update}
    stamps = worker_stamps(
        profile_name=stamp_name,
        profile_revision=stamp_revision,
        pack_stamp=verify.pack_stamp if verify else None,
        vlm_called=verify is not None and probe is not None,
        vlm_endpoint=verify.labeler.identity.endpoint_ref if verify else None,
        vlm_model=verify.labeler.identity.model if verify else None,
    )
    write = finalize_region_write(
        update,
        source,
        doc_id=crop_id,
        class_token=class_state_token(source),
        trace=trace,
        stamps=stamps,
    )
    return RegionRun(
        legs=legs, preview=preview_item(source, write, crop_id), basis=basis, probe=probe
    )


__all__ = ['Leg', 'RegionRun', 'VerifyContext', 'run_region_test']

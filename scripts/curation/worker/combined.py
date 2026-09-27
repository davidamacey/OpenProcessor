"""Combined class + region-verify + OCR helpers for the curation worker.

Extracted from ``cascade.py`` so the cascade module stays under the
700-LOC ceiling. Houses:

* ``_is_combined_cohort`` — which ``class_source`` values (derived from
  the configured ingest profiles) trigger the combined VLM call.
* ``_try_combined_class_region`` — the per-crop helper that runs the
  single VLM round-trip and (on success) writes a fully-formed
  ``update_doc``. Always stashes the class-side update on
  ``task.combined_class_update`` so downstream region writes can layer
  it in without clobbering.
* ``_finalize_no_region`` — terminal write helper used by the cascade
  when every region detector misses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.curation.worker.no_verdict import cascade_no_verdict
from scripts.curation.worker.state import (
    _PENDING_DETECTION_ALIASES,
    _PENDING_VERIFICATION_ALIASES,
    _is_secondary_shape,
    _ItemTask,
    region_profile,
)
from scripts.curation.worker.verify import (
    _auto_confirm_or_pending,
    _combined_class_update,
    _combined_write_doc,
)
from src.config import get_region_fields
from src.config.region_source import (
    CANDIDATE_DETECTOR,
    CANDIDATE_DETECTOR_EXISTING,
    CANDIDATE_SEGMENTER,
)
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.ingest_class_sources import (
    CLUSTER_MAJORITY_CLASS_SOURCE,
    classifier_class_sources,
    unlabeled_proposal_class_sources,
)
from src.services.detection.cascade_detect import (
    RegionDetector,
    crop_norm_to_source_norm,
    is_plausible_region_bbox,
)
from src.services.labeling.vlm_labeler import CombinedParseFailure


if TYPE_CHECKING:
    from scripts.curation.worker.cascade import SegmenterClient
    from src.services.labeling.vlm_labeler import VlmLabeler


logger = get_logger('curation_worker')


# Cohort markers come from the configured ingest profiles
# (src.services.curation.ingest_class_sources): an item the ingest
# detectors left unlabeled (a primary proposal / low-conf box) still needs
# a class, so class + region-verify + OCR roll into one VLM call.
#
# Phase C: also low-confidence classifier / cluster-majority crops, so
# they get class + region-verify + OCR in ONE VLM call instead of TWO.
# The threshold mirrors the pipeline's VLM-skip confidence default (0.80)
# -- crops at or above it are trusted enough that re-asking adds nothing.
_CLASSIFIER_LOW_CONF_THRESHOLD = 0.80


def _is_combined_cohort(class_source: str, class_confidence: float) -> bool:
    """Return True when this crop should take the combined VLM call path.

    Cohort rule:
      * ``class_source`` is an unlabeled ingest proposal
        (``unlabeled_proposal_class_sources()``) -- no classifier fired, OR
      * ``class_source`` is a configured classifier source or the
        cluster-majority source AND confidence < 0.80.

    High-confidence classifier crops fall through to the legacy two-call
    path because the class is already trustworthy.
    """
    if class_source in unlabeled_proposal_class_sources():
        return True
    low_conf_sources = classifier_class_sources() | {CLUSTER_MAJORITY_CLASS_SOURCE}
    return class_source in low_conf_sources and class_confidence < _CLASSIFIER_LOW_CONF_THRESHOLD


def _finalize_no_region(task: _ItemTask) -> None:
    """Write the terminal no_region_box doc, layering combined class fields."""
    F = get_region_fields()
    task.update_doc = {F.status: RegionStatus.NO_REGION_BOX}
    if task.detection_trace:
        task.update_doc[F.detector_chain] = list(task.detection_trace)
    if task.combined_class_update:
        task.update_doc.update(task.combined_class_update)


async def _try_combined_class_region(
    task: _ItemTask,
    *,
    candidate_in_crop: tuple[float, float, float, float],
    candidate_in_source: tuple[float, float, float, float],
    candidate_score: float,
    detector: str,
    detector_version: str,
    detector_chain_tag: str,
    vlm: VlmLabeler,
    candidate_source: str = CANDIDATE_DETECTOR,
) -> bool:
    """Run the primary-detector-missed combined VLM call. Returns True on success.

    On success populates ``task.update_doc`` with class fields, region
    fields, and ``vlm_verify_completed_at`` so downstream pipelines
    know class+region were resolved in one round-trip. On parse failure
    returns False and the caller falls back to the legacy two-call path.
    A reply with no verdict on the box also returns True, with an empty
    ``update_doc``: nothing is written and the item stays pending -- until
    the no-verdict cap, when the candidate is parked for review instead.
    """
    class_names = getattr(vlm, 'class_names', None) or []
    name_to_id = getattr(vlm, 'name_to_id', None) or {}
    try:
        reply = await vlm.label_combined(
            img_id=task.crop_id,
            jpeg_bytes=task.crop_jpeg or b'',
            class_names=list(class_names),
            region_bbox_norm=candidate_in_crop,
            draw_overlay=True,
        )
    except CombinedParseFailure as exc:
        logger.info('curation_combined_parse_failure', crop_id=task.crop_id, error=str(exc))
        return False

    # Stash class-side update so the eventual no_region_box terminal write
    # can layer it in (via _finalize_no_region) when the region-side
    # verification fails.
    task.combined_class_update = _combined_class_update(
        reply, list(class_names), name_to_id=name_to_id
    )

    if reply.region_visible and reply.region_bbox_correct is None:
        # A visible region but no verdict on the box (null / absent): not a
        # reject. Resolve the crop with no write so it stays pending and the
        # next pass retries it.
        task.detection_trace.append(f'{detector_chain_tag}:combined_no_verdict')
        cascade_no_verdict(
            task,
            actor=detector,
            detector_version=detector_version,
            candidate_in_source=candidate_in_source,
            candidate_score=candidate_score,
            candidate_source=candidate_source,
            event='combined_verify_reject',
            class_update=task.combined_class_update,
        )
        return True

    if reply.region_bbox_correct and reply.region_visible:
        gate_ok, gate_reason = is_plausible_region_bbox(candidate_in_crop, task.item_bbox_norm)
        if gate_ok:
            task.detection_trace.append(f'{detector_chain_tag}:hit')
            task.detection_trace.append(f'{detector_chain_tag}:combined_verify_ok')
            high_conf = reply.region_confidence == 'high'
            auto = await _auto_confirm_or_pending(
                sam_score=candidate_score,
                bbox_in_crop=candidate_in_crop,
                vlm_high_conf=high_conf,
            )
            task.update_doc = _combined_write_doc(
                reply=reply,
                candidate_in_source=candidate_in_source,
                candidate_score=candidate_score,
                detector=detector,
                detector_version=detector_version,
                chain=task.detection_trace,
                class_names=list(class_names),
                auto_confirmed=bool(auto),
                name_to_id=name_to_id,
            )
            return True
        task.detection_trace.append(f'{detector_chain_tag}:sanity_reject:{gate_reason}')
    task.detection_trace.append(f'{detector_chain_tag}:combined_verify_reject')
    return False


async def _try_combined_on_segmenter(
    task: _ItemTask,
    *,
    segmenter: SegmenterClient,
    vlm: VlmLabeler,
) -> bool:
    """Run the secondary segmenter on the crop, then a combined VLM call.

    Returns True if a final ``update_doc`` was written (either via the
    combined call or the terminal no_region_box helper); False means the
    caller should keep going through legacy paths.
    """
    cand = await segmenter.segment(task.crop_jpeg or b'')
    if cand is None:
        task.detection_trace.append(f'{region_profile().segmenter_name}:miss')
        _finalize_no_region(task)
        return True
    gate_ok, gate_reason = is_plausible_region_bbox(cand.bbox_norm, task.item_bbox_norm)
    if not gate_ok:
        task.detection_trace.append(
            f'{region_profile().segmenter_name}:sanity_reject:{gate_reason}'
        )
        _finalize_no_region(task)
        return True
    projected = crop_norm_to_source_norm(cand.bbox_norm, task.item_bbox_norm)
    ok = await _try_combined_class_region(
        task,
        candidate_in_crop=cand.bbox_norm,
        candidate_in_source=projected,
        candidate_score=cand.score,
        detector=region_profile().segmenter_name,
        detector_version=region_profile().segmenter_version,
        detector_chain_tag=region_profile().segmenter_name,
        vlm=vlm,
        candidate_source=CANDIDATE_SEGMENTER,
    )
    if ok:
        return True
    _finalize_no_region(task)
    return True


async def _run_combined_cohort_path(
    task: _ItemTask,
    *,
    detector: RegionDetector,
    segmenter: SegmenterClient,
    vlm: VlmLabeler,
) -> bool:
    """Route a primary-detector-missed cohort crop through the combined VLM path.

    Returns True when this function has fully resolved the crop (caller
    should ``return``); False means the cohort path didn't apply or the
    combined call indicated "use the legacy secondary-segmenter fallback".
    """
    if not _is_combined_cohort(task.class_source, task.class_confidence):
        return False
    is_secondary = _is_secondary_shape(task)

    if (
        task.region_status in _PENDING_VERIFICATION_ALIASES
        and task.detector_region_in_source is not None
    ):
        from scripts.curation.worker.cascade import _source_to_crop  # avoid import cycle

        cand_in_crop = _source_to_crop(task.detector_region_in_source, task.item_bbox_norm)
        # Combined success → True. Bbox-wrong → False so the legacy
        # secondary-segmenter cascade runs and tries to find a
        # different region bbox.
        return await _try_combined_class_region(
            task,
            candidate_in_crop=cand_in_crop,
            candidate_in_source=task.detector_region_in_source,
            candidate_score=task.detector_score,
            detector=region_profile().detector_model,
            detector_version=region_profile().detector_version,
            detector_chain_tag=region_profile().detector_model,
            vlm=vlm,
            candidate_source=CANDIDATE_DETECTOR_EXISTING,
        )

    if task.region_status in _PENDING_DETECTION_ALIASES and not is_secondary:
        if not region_profile().detector_model:
            # No detector leg: the segmenter proposes the only candidate.
            return await _try_combined_on_segmenter(task, segmenter=segmenter, vlm=vlm)
        return await _run_combined_pending_detection(
            task, detector=detector, segmenter=segmenter, vlm=vlm
        )

    return False


async def _run_combined_pending_detection(
    task: _ItemTask,
    *,
    detector: RegionDetector,
    segmenter: SegmenterClient,
    vlm: VlmLabeler,
) -> bool:
    """pending_detection cohort branch — run the primary detector, then
    combined; secondary-segmenter fallback."""
    if task.crop_jpeg is None:
        return False
    detector_results = await detector.detect_batch([task.crop_jpeg])
    cand = detector_results[0] if detector_results else None
    if cand is None:
        task.detection_trace.append(f'{region_profile().detector_model}:miss')
        return await _try_combined_on_segmenter(task, segmenter=segmenter, vlm=vlm)
    gate_ok, gate_reason = is_plausible_region_bbox(cand.bbox_norm, task.item_bbox_norm)
    if not gate_ok:
        task.detection_trace.append(f'{region_profile().detector_model}:hit')
        task.detection_trace.append(
            f'{region_profile().detector_model}:sanity_reject:{gate_reason}'
        )
        # Fall through to legacy cascade (secondary-segmenter path with
        # trace already populated).
        return False
    projected = crop_norm_to_source_norm(cand.bbox_norm, task.item_bbox_norm)
    ok = await _try_combined_class_region(
        task,
        candidate_in_crop=cand.bbox_norm,
        candidate_in_source=projected,
        candidate_score=cand.score,
        detector=region_profile().detector_model,
        detector_version=region_profile().detector_version,
        detector_chain_tag=region_profile().detector_model,
        vlm=vlm,
        candidate_source=CANDIDATE_DETECTOR,
    )
    if ok:
        return True
    # Primary-detector bbox combined rejected; try secondary segmenter + combined.
    return await _try_combined_on_segmenter(task, segmenter=segmenter, vlm=vlm)

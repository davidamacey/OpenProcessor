"""B-PR5 combined class + region-verify + OCR helpers for the curation worker.

Extracted from ``cascade.py`` so the cascade module stays under the
700-LOC ceiling (PR7 §1 / §2). Houses:

* ``COHORT_V6_MISSED`` — the ``class_source`` marker that triggers the
  combined VLM call.
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

from scripts.curation.worker.state import (
    _PENDING_DETECTION_ALIASES,
    _PENDING_VERIFICATION_ALIASES,
    _is_secondary_shape,
    _ItemTask,
)
from scripts.curation.worker.verify import (
    _auto_confirm_or_pending,
    _combined_class_update,
    _combined_write_doc,
)
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.detection.cascade_detect import (
    REFERENCE_LICENSE_PLATE_PROFILE,
    RegionDetector,
    crop_norm_to_source_norm,
    is_plausible_region_bbox,
)
from src.services.labeling.vlm_labeler import CombinedParseFailure


if TYPE_CHECKING:
    from scripts.curation.worker.cascade import Sam3Client
    from src.services.labeling.vlm_labeler import VlmLabeler


logger = get_logger('curation_worker')


# Cohort marker on the items index. ``coco_yolo11_proposal`` is written
# by ingest when the primary classifier misses and we fall back to a
# generic proposal. Those crops still need a class label; rolling class
# + region-verify + OCR into a single VLM call cuts ~2 round-trips per
# crop.
COHORT_V6_MISSED = 'coco_yolo11_proposal'

# Phase C: broaden the cohort to include low-confidence primary /
# cluster-primary crops so we get class + region-verify + OCR in ONE
# VLM call instead of TWO (one for class, one for region-verify). The
# threshold mirrors the pipeline's ``classifier_confidence_skip_vlm`` default
# (0.80) — crops at or above that confidence are trusted enough that
# re-asking the VLM adds no signal, so the legacy two-call path runs.
_LOW_CONF_CLASS_SOURCES: frozenset[str] = frozenset({'item_model', 'cluster_majority_agreement'})
_V6_LOW_CONF_THRESHOLD = 0.80


def _is_combined_cohort(class_source: str, class_confidence: float) -> bool:
    """Return True when this crop should take the combined VLM call path.

    Cohort rule (Phase C):
      * ``class_source == 'coco_yolo11_proposal'`` (the original B-PR5
        cohort — the primary classifier missed entirely) OR
      * ``class_source`` in {'item_model', 'cluster_majority_agreement'}
        AND confidence < 0.80 (the primary classifier fired but with
        low confidence, so the class still needs VLM clarification).

    High-confidence primary-classifier crops fall through to the legacy
    two-call path because the class is already trustworthy; combining
    would waste the larger VLM prompt budget on a crop whose only open
    question is region verification.
    """
    if class_source == COHORT_V6_MISSED:
        return True
    return class_source in _LOW_CONF_CLASS_SOURCES and class_confidence < _V6_LOW_CONF_THRESHOLD


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
    gemma: VlmLabeler,
) -> bool:
    """Run the primary-detector-missed combined VLM call. Returns True on success.

    On success populates ``task.update_doc`` with class fields, region
    fields, and ``vlm_verify_completed_at`` so downstream pipelines
    know class+region were resolved in one round-trip. On parse failure
    returns False and the caller falls back to the legacy two-call path.
    """
    class_names = getattr(gemma, 'class_names', None) or []
    name_to_id = getattr(gemma, 'name_to_id', None) or {}
    try:
        reply = await gemma.label_combined(
            img_id=task.crop_id,
            jpeg_bytes=task.crop_jpeg or b'',
            class_names=list(class_names),
            plate_bbox_norm=candidate_in_crop,
            draw_overlay=True,
        )
    except CombinedParseFailure as exc:
        logger.info('legacy_combined_parse_failure', crop_id=task.crop_id, error=str(exc))
        return False

    # Stash class-side update so the eventual no_region_box terminal write
    # can layer it in (via _finalize_no_region) when the region-side
    # verification fails.
    task.combined_class_update = _combined_class_update(
        reply, list(class_names), name_to_id=name_to_id
    )

    if reply.plate_bbox_correct and reply.plate_visible:
        gate_ok, gate_reason = is_plausible_region_bbox(candidate_in_crop, task.vehicle_bbox_norm)
        if gate_ok:
            task.detection_trace.append(f'{detector_chain_tag}:hit')
            task.detection_trace.append(f'{detector_chain_tag}:combined_verify_ok')
            high_conf = reply.plate_confidence == 'high'
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
                plate_validated=auto or False,
                name_to_id=name_to_id,
            )
            return True
        task.detection_trace.append(f'{detector_chain_tag}:sanity_reject:{gate_reason}')
    task.detection_trace.append(f'{detector_chain_tag}:combined_verify_reject')
    return False


async def _try_combined_on_sam3(
    task: _ItemTask,
    *,
    sam3: Sam3Client,
    gemma: VlmLabeler,
) -> bool:
    """Run the secondary segmenter on the crop, then a combined VLM call.

    Returns True if a final ``update_doc`` was written (either via the
    combined call or the terminal no_region_box helper); False means the
    caller should keep going through legacy paths.
    """
    cand = await sam3.segment_plate(task.crop_jpeg or b'')
    if cand is None:
        task.detection_trace.append(f'{REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name}:miss')
        _finalize_no_region(task)
        return True
    gate_ok, gate_reason = is_plausible_region_bbox(cand.bbox_norm, task.vehicle_bbox_norm)
    if not gate_ok:
        task.detection_trace.append(
            f'{REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name}:sanity_reject:{gate_reason}'
        )
        _finalize_no_region(task)
        return True
    projected = crop_norm_to_source_norm(cand.bbox_norm, task.vehicle_bbox_norm)
    ok = await _try_combined_class_region(
        task,
        candidate_in_crop=cand.bbox_norm,
        candidate_in_source=projected,
        candidate_score=cand.score,
        detector=REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name,
        detector_version=REFERENCE_LICENSE_PLATE_PROFILE.segmenter_version,
        detector_chain_tag=REFERENCE_LICENSE_PLATE_PROFILE.segmenter_name,
        gemma=gemma,
    )
    if ok:
        return True
    _finalize_no_region(task)
    return True


async def _run_combined_cohort_path(
    task: _ItemTask,
    *,
    lpr: RegionDetector,
    sam3: Sam3Client,
    gemma: VlmLabeler,
) -> bool:
    """Route a primary-detector-missed cohort crop through the combined VLM path.

    Returns True when this function has fully resolved the crop (caller
    should ``return``); False means the cohort path didn't apply or the
    combined call indicated "use the legacy secondary-segmenter fallback".
    """
    if not _is_combined_cohort(task.class_source, task.class_confidence):
        return False
    is_secondary = _is_secondary_shape(task)

    if task.plate_status in _PENDING_VERIFICATION_ALIASES and task.lpr_plate_in_source is not None:
        from scripts.curation.worker.cascade import _source_to_crop  # avoid import cycle

        cand_in_crop = _source_to_crop(task.lpr_plate_in_source, task.vehicle_bbox_norm)
        # Combined success → True. Bbox-wrong → False so the legacy
        # secondary-segmenter cascade runs and tries to find a
        # different region bbox.
        return await _try_combined_class_region(
            task,
            candidate_in_crop=cand_in_crop,
            candidate_in_source=task.lpr_plate_in_source,
            candidate_score=task.lpr_score,
            detector=REFERENCE_LICENSE_PLATE_PROFILE.detector_model,
            detector_version=REFERENCE_LICENSE_PLATE_PROFILE.detector_version,
            detector_chain_tag=REFERENCE_LICENSE_PLATE_PROFILE.detector_model,
            gemma=gemma,
        )

    if task.plate_status in _PENDING_DETECTION_ALIASES and not is_secondary:
        return await _run_combined_pending_detection(task, lpr=lpr, sam3=sam3, gemma=gemma)

    return False


async def _run_combined_pending_detection(
    task: _ItemTask,
    *,
    lpr: RegionDetector,
    sam3: Sam3Client,
    gemma: VlmLabeler,
) -> bool:
    """pending_detection cohort branch — run the primary detector, then
    combined; secondary-segmenter fallback."""
    if task.crop_jpeg is None:
        return False
    lpr_results = await lpr.detect_batch([task.crop_jpeg])
    cand = lpr_results[0] if lpr_results else None
    if cand is None:
        task.detection_trace.append(f'{REFERENCE_LICENSE_PLATE_PROFILE.detector_model}:miss')
        return await _try_combined_on_sam3(task, sam3=sam3, gemma=gemma)
    gate_ok, gate_reason = is_plausible_region_bbox(cand.bbox_norm, task.vehicle_bbox_norm)
    if not gate_ok:
        task.detection_trace.append(f'{REFERENCE_LICENSE_PLATE_PROFILE.detector_model}:hit')
        task.detection_trace.append(
            f'{REFERENCE_LICENSE_PLATE_PROFILE.detector_model}:sanity_reject:{gate_reason}'
        )
        # Fall through to legacy cascade (secondary-segmenter path with
        # trace already populated).
        return False
    projected = crop_norm_to_source_norm(cand.bbox_norm, task.vehicle_bbox_norm)
    ok = await _try_combined_class_region(
        task,
        candidate_in_crop=cand.bbox_norm,
        candidate_in_source=projected,
        candidate_score=cand.score,
        detector=REFERENCE_LICENSE_PLATE_PROFILE.detector_model,
        detector_version=REFERENCE_LICENSE_PLATE_PROFILE.detector_version,
        detector_chain_tag=REFERENCE_LICENSE_PLATE_PROFILE.detector_model,
        gemma=gemma,
    )
    if ok:
        return True
    # Primary-detector bbox combined rejected; try secondary segmenter + combined.
    return await _try_combined_on_sam3(task, sam3=sam3, gemma=gemma)

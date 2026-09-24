"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/sam_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

# ruff: noqa: E402
import base64  # noqa: F401  — kept for back-compat re-export surface
import io
from typing import TYPE_CHECKING, Any

from PIL import Image

from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.detection.cascade_detect import (
    PaddleOcrTextRecognizer,
    RegionCandidate,
    RegionDetector,
    crop_norm_to_source_norm,
    is_plausible_region_bbox,
)


logger = get_logger('curation_worker')


from scripts.curation.worker.client import (
    Sam3AllHostsDown,  # noqa: F401  # back-compat re-export for runner/tests
    Sam3Client,  # noqa: TC001  # runtime back-compat re-export for shim + tests
)
from scripts.curation.worker.combined import _finalize_no_region, _run_combined_cohort_path
from scripts.curation.worker.state import (
    _PENDING_DETECTION_ALIASES,
    _PENDING_VERIFICATION_ALIASES,
    _TERMINAL_STATUSES,
    CURATION_ITEMS_INDEX,
    JPEG_QUALITY,
    _is_secondary_shape,
    _ItemTask,
    region_profile,
)
from scripts.curation.worker.verify import (
    _SKIP_VLM_VERIFY_SECONDARY_SCORE,
    _auto_confirm_or_pending,
    _bbox_shape_is_plausible,
    _region_write_doc,
    _verify_with_vlm,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.labeling.vlm_labeler import VlmLabeler


def _build_pending_query() -> dict[str, Any]:
    """Crops needing the worker's attention — pending or pending_verify.

    Skips crops whose region status is already terminal so we never
    overwrite a human or detector verdict on a re-run.

    Deliberately **not** filtered on ``test_holdout`` (P0-3): this query
    feeds the whole worker pipeline, which writes region fields
    unconditionally for every fetched crop. Excluding holdout crops here
    would silently starve them of region detection too, violating the
    region-fields-stay-unconditional rule. The test_holdout guard for
    this worker is scoped to the class-field write path only — see
    ``runner.py:_should_classify`` (checks ``task.test_holdout``).
    """
    F = get_region_fields()
    return {
        'bool': {
            'must': [
                {'exists': {'field': 'image_path'}},
                {'exists': {'field': 'bbox_norm'}},
                # Pull both legacy short names AND the renamed forms
                # introduced by task #7. Compatibility window: worker
                # consumes whichever name OS happens to carry today;
                # writes only ever emit the new long-form names below.
                {
                    'terms': {
                        F.status: [
                            'pending',
                            RegionStatus.PENDING_DETECTION,
                            'pending_verify',
                            RegionStatus.PENDING_VERIFICATION,
                        ]
                    }
                },
            ],
        },
    }


async def _fetch_pending(opensearch: AsyncOpenSearch, *, batch_size: int) -> list[_ItemTask]:
    """Pull up to ``batch_size`` pending crops, oldest first."""
    F = get_region_fields()
    body = {
        'size': batch_size,
        '_source': [
            'crop_id',
            'image_path',
            'bbox_norm',
            F.status,
            F.bbox_norm,
            F.score,
            'class_name',
            'class_source',
            'class_validated',
            'confidence',
            'request_id',
            'test_holdout',
        ],
        'query': _build_pending_query(),
        'sort': [{'created_at': {'order': 'asc', 'unmapped_type': 'date'}}],
    }
    resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    hits = (resp.get('hits') or {}).get('hits') or []
    tasks: list[_ItemTask] = []
    for h in hits:
        src = h.get('_source') or {}
        bbox = src.get('bbox_norm')
        if not bbox or len(bbox) != 4:
            continue
        plate_bbox = src.get(F.bbox_norm)
        lpr_in_source: tuple[float, float, float, float] | None = None
        if isinstance(plate_bbox, list) and len(plate_bbox) == 4:
            lpr_in_source = (
                float(plate_bbox[0]),
                float(plate_bbox[1]),
                float(plate_bbox[2]),
                float(plate_bbox[3]),
            )
        tasks.append(
            _ItemTask(
                crop_id=h['_id'],
                image_path=str(src.get('image_path') or ''),
                vehicle_bbox_norm=(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
                plate_status=src.get(F.status),
                class_name=str(src.get('class_name') or ''),
                class_source=str(src.get('class_source') or ''),
                class_confidence=float(src.get('confidence') or 0.0),
                class_validated=bool(src.get('class_validated') or False),
                test_holdout=bool(src.get('test_holdout') or False),
                lpr_plate_in_source=lpr_in_source,
                lpr_score=float(src.get(F.score) or 0.0),
                request_id=str(src.get('request_id') or '-'),
            )
        )
    return tasks


# =============================================================================
# Per-crop pipeline
# =============================================================================


def _crop_region_jpeg(crop_jpeg: bytes, plate_in_crop: tuple[float, float, float, float]) -> bytes:
    """Extract just the region of interest from a crop's JPEG bytes for VLM verify."""
    img = Image.open(io.BytesIO(crop_jpeg))
    img.load()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    cw, ch = img.size
    px1, py1, px2, py2 = plate_in_crop
    x1 = max(0, round(px1 * cw))
    y1 = max(0, round(py1 * ch))
    x2 = max(x1 + 1, round(px2 * cw))
    y2 = max(y1 + 1, round(py2 * ch))
    plate = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    plate.save(buf, format='JPEG', quality=JPEG_QUALITY)
    return buf.getvalue()


def _source_to_crop(
    plate_in_source: tuple[float, float, float, float],
    vehicle_in_source: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Inverse of :func:`crop_norm_to_source_norm` — needed to verify a
    primary-detector region that was already projected into source
    coordinates.

    Used only for ``pending_verify`` crops where the upstream ingest
    wrote the region bbox in source frame; the VLM needs a tight region
    JPEG, which we can only carve from the item crop.
    """
    vx1, vy1, vx2, vy2 = vehicle_in_source
    vw = max(vx2 - vx1, 1e-6)
    vh = max(vy2 - vy1, 1e-6)
    sx1, sy1, sx2, sy2 = plate_in_source
    return (
        max(0.0, min(1.0, (sx1 - vx1) / vw)),
        max(0.0, min(1.0, (sy1 - vy1) / vh)),
        max(0.0, min(1.0, (sx2 - vx1) / vw)),
        max(0.0, min(1.0, (sy2 - vy1) / vh)),
    )


# Sub-crop margin around an OCR text hint before re-running the
# secondary segmenter. The segmenter works better with a tighter view
# of the text-bearing region than with the full item crop. 1.8x bounds
# the neighborhood so the region keeps some context (mounting bracket,
# body-color edge) which the grounding model uses to disambiguate
# badges from the real region of interest.
_TEXT_HINT_SUBCROP_MARGIN = 1.8


def _expand_bbox(
    bbox_in_crop: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    """Return an axis-aligned expansion of ``bbox_in_crop`` by ``margin``.

    ``margin=1.8`` means the resulting box is 1.8x the original size,
    centered on the same point, clipped to ``[0, 1]``.
    """
    x1, y1, x2, y2 = bbox_in_crop
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    hw = (x2 - x1) * 0.5 * margin
    hh = (y2 - y1) * 0.5 * margin
    return (
        max(0.0, cx - hw),
        max(0.0, cy - hh),
        min(1.0, cx + hw),
        min(1.0, cy + hh),
    )


def _project_subcrop_box_to_parent(
    box_in_sub: tuple[float, float, float, float],
    sub_in_parent: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Project a sub-crop-frame bbox back into its parent-crop frame."""
    sx1, sy1, sx2, sy2 = sub_in_parent
    sw = max(sx2 - sx1, 1e-6)
    sh = max(sy2 - sy1, 1e-6)
    bx1, by1, bx2, by2 = box_in_sub
    return (
        max(0.0, min(1.0, sx1 + bx1 * sw)),
        max(0.0, min(1.0, sy1 + by1 * sh)),
        max(0.0, min(1.0, sx1 + bx2 * sw)),
        max(0.0, min(1.0, sy1 + by2 * sh)),
    )


async def _resegment_from_text_hint(
    crop_jpeg: bytes,
    hint_in_crop: tuple[float, float, float, float],
    sam3: Sam3Client,
) -> tuple[RegionCandidate | None, tuple[float, float, float, float]]:
    """Run the secondary segmenter on a tight sub-crop around an OCR text hint.

    Returns ``(candidate_in_parent_crop, sub_in_parent)``. The candidate
    is projected back to the parent-crop frame so the rest of the worker
    treats it identically to a global hit. ``sub_in_parent`` is
    surfaced for trace logging.
    """
    sub_box = _expand_bbox(hint_in_crop, _TEXT_HINT_SUBCROP_MARGIN)
    sub_jpeg = _crop_region_jpeg(crop_jpeg, sub_box)
    sub_cand = await sam3.segment_plate(sub_jpeg)
    if sub_cand is None:
        return None, sub_box
    projected = _project_subcrop_box_to_parent(sub_cand.bbox_norm, sub_box)
    return (
        RegionCandidate(bbox_norm=projected, score=sub_cand.score),
        sub_box,
    )


async def _process_crop(
    task: _ItemTask,
    *,
    lpr: RegionDetector,
    sam3: Sam3Client,
    ocr_recognizer: PaddleOcrTextRecognizer,
    gemma: VlmLabeler,
) -> None:
    """Run the routing logic for one crop and populate ``task.update_doc``.

    See module docstring for the routing rules.
    """
    if task.plate_status in _TERMINAL_STATUSES:
        # Defensive — query already filters these out.
        return

    if task.crop_jpeg is None:
        # Cropping failed at fetch time. Leave the doc alone — this is
        # almost always a transient infra problem (missing volume mount,
        # storage unmounted, broken JPEG) and writing a terminal status
        # here would lock the crop out of the detection workflow forever
        # once the underlying issue is fixed. The worker logs the reason
        # and the next iteration will retry.
        return

    is_secondary = _is_secondary_shape(task)
    sam_candidate: RegionCandidate | None = None
    det_model = region_profile().detector_model
    det_version = region_profile().detector_version
    seg_name = region_profile().segmenter_name
    seg_version = region_profile().segmenter_version
    ocr_det_model = region_profile().ocr_rec_model

    # ---- Step 0: B-PR5 combined class+region for low-confidence-class cohort. ----
    # Cohort (Phase C broadened) = ``class_source='coco_yolo11_proposal'``
    # (primary classifier missed) OR ``class_source in {'item_model',
    # 'cluster_majority_agreement'}`` with confidence < 0.80, AND a
    # region candidate exists or can be cheaply produced. One VLM call
    # returns class + region verify + OCR instead of two/three round-trips.
    # Implementation lives in ``combined._run_combined_cohort_path``.
    if await _run_combined_cohort_path(task, lpr=lpr, sam3=sam3, gemma=gemma):
        return
    # Non-cohort or cohort fell back — legacy cascade resumes.

    # ---- Step 1: pending_verify path. ----
    if task.plate_status in _PENDING_VERIFICATION_ALIASES and task.lpr_plate_in_source is not None:
        plate_in_crop = _source_to_crop(task.lpr_plate_in_source, task.vehicle_bbox_norm)
        plate_jpeg = _crop_region_jpeg(task.crop_jpeg, plate_in_crop)
        outcome = await _verify_with_vlm(gemma, task.crop_id, plate_jpeg)
        ok, conf = outcome.ok, outcome.confidence
        if ok:
            # Phase A3 sanity gate. The pending_verify path's region came
            # from an earlier primary-detector ingest; re-check before
            # committing the verified write. On reject, record the
            # detector + reason and fall through to the secondary
            # segmenter anyway (the trace captures both).
            gate_ok, gate_reason = is_plausible_region_bbox(plate_in_crop, task.vehicle_bbox_norm)
            if not gate_ok:
                task.detection_trace.append(f'{det_model}:sanity_reject:{gate_reason}')
                # Fall through to the secondary segmenter.
            else:
                auto = await _auto_confirm_or_pending(
                    sam_score=task.lpr_score,
                    bbox_in_crop=plate_in_crop,
                    vlm_high_conf=conf == 'high',
                )
                task.detection_trace.append(f'{det_model}:hit')
                task.detection_trace.append(f'{det_model}:vlm_verify_ok')
                task.update_doc = _region_write_doc(
                    plate_in_source=task.lpr_plate_in_source,
                    score=task.lpr_score,
                    detector=det_model,
                    detector_version=det_version,
                    chain=task.detection_trace,
                    plate_validated=auto or False,
                    plate_text=outcome.text,
                    plate_text_confidence=outcome.text_confidence,
                )
                return
        else:
            task.detection_trace.append(f'{det_model}:vlm_reject')
        # Verify rejected — fall through to the secondary segmenter.

    # ---- Step 2: primary detector (only if pending + non-secondary-shape). ----
    elif task.plate_status in _PENDING_DETECTION_ALIASES and not is_secondary:
        lpr_results = await lpr.detect_batch([task.crop_jpeg])
        cand = lpr_results[0] if lpr_results else None
        if cand is None:
            task.detection_trace.append(f'{det_model}:miss')
        else:
            # Phase A3 sanity gate on the fresh primary-detector candidate.
            gate_ok, gate_reason = is_plausible_region_bbox(cand.bbox_norm, task.vehicle_bbox_norm)
            if not gate_ok:
                task.detection_trace.append(f'{det_model}:hit')
                task.detection_trace.append(f'{det_model}:sanity_reject:{gate_reason}')
                # Fall through to the secondary segmenter.
            else:
                plate_jpeg = _crop_region_jpeg(task.crop_jpeg, cand.bbox_norm)
                outcome = await _verify_with_vlm(gemma, task.crop_id, plate_jpeg)
                ok, conf = outcome.ok, outcome.confidence
                if ok:
                    projected = crop_norm_to_source_norm(cand.bbox_norm, task.vehicle_bbox_norm)
                    auto = await _auto_confirm_or_pending(
                        sam_score=cand.score,
                        bbox_in_crop=cand.bbox_norm,
                        vlm_high_conf=conf == 'high',
                    )
                    task.detection_trace.append(f'{det_model}:hit')
                    task.detection_trace.append(f'{det_model}:vlm_verify_ok')
                    task.update_doc = _region_write_doc(
                        plate_in_source=projected,
                        score=cand.score,
                        detector=det_model,
                        detector_version=det_version,
                        chain=task.detection_trace,
                        plate_validated=auto or False,
                        plate_text=outcome.text,
                        plate_text_confidence=outcome.text_confidence,
                    )
                    return
                task.detection_trace.append(f'{det_model}:hit')
                task.detection_trace.append(f'{det_model}:vlm_reject')
                # Primary detector hit but VLM rejected — fall through
                # to the secondary segmenter.

    # ---- Step 3: secondary segmenter (always — secondary-shape pending,
    #              non-secondary primary-detector miss/reject, or
    #              pending_verify reject). ----
    sam_candidate = await sam3.segment_plate(task.crop_jpeg)
    if sam_candidate is None:
        task.detection_trace.append(f'{seg_name}:miss')
    else:
        # Phase A3 sanity gate. Reject early before the VLM roundtrip.
        gate_ok, gate_reason = is_plausible_region_bbox(
            sam_candidate.bbox_norm, task.vehicle_bbox_norm
        )
        if not gate_ok:
            task.detection_trace.append(f'{seg_name}:hit')
            task.detection_trace.append(f'{seg_name}:sanity_reject:{gate_reason}')
            # Fall through to text-hint (OCR-hinted secondary-segmenter re-pass).
        else:
            # Fast path: skip VLM verify when the secondary segmenter is
            # very confident AND the bbox shape passes the region
            # sanity check.
            if sam_candidate.score >= _SKIP_VLM_VERIFY_SECONDARY_SCORE and _bbox_shape_is_plausible(
                sam_candidate.bbox_norm
            ):
                projected = crop_norm_to_source_norm(
                    sam_candidate.bbox_norm, task.vehicle_bbox_norm
                )
                task.detection_trace.append(f'{seg_name}:hit')
                task.detection_trace.append(f'{seg_name}:skip_vlm_verify')
                task.update_doc = _region_write_doc(
                    plate_in_source=projected,
                    score=sam_candidate.score,
                    detector=seg_name,
                    detector_version=seg_version,
                    chain=task.detection_trace,
                    plate_verified=False,
                    plate_validated=False,
                    verifier=None,
                    verifier_version=None,
                    extra={get_region_fields().skip_verify: True},
                )
                return

            plate_jpeg = _crop_region_jpeg(task.crop_jpeg, sam_candidate.bbox_norm)
            outcome = await _verify_with_vlm(gemma, task.crop_id, plate_jpeg)
            ok, conf = outcome.ok, outcome.confidence
            if ok:
                projected = crop_norm_to_source_norm(
                    sam_candidate.bbox_norm, task.vehicle_bbox_norm
                )
                auto = await _auto_confirm_or_pending(
                    sam_score=sam_candidate.score,
                    bbox_in_crop=sam_candidate.bbox_norm,
                    vlm_high_conf=conf == 'high',
                )
                task.detection_trace.append(f'{seg_name}:hit')
                task.detection_trace.append(f'{seg_name}:vlm_verify_ok')
                task.update_doc = _region_write_doc(
                    plate_in_source=projected,
                    score=sam_candidate.score,
                    detector=seg_name,
                    detector_version=seg_version,
                    chain=task.detection_trace,
                    plate_validated=auto or False,
                    plate_text=outcome.text,
                    plate_text_confidence=outcome.text_confidence,
                )
                return
            task.detection_trace.append(f'{seg_name}:hit')
            task.detection_trace.append(f'{seg_name}:vlm_reject')

    # ---- Step 4: text-hint-driven secondary-segmenter re-pass. ----
    # The OCR-detection model is no longer trusted as a region-bbox
    # source (its text-detect regions are too loose and produced
    # oversized boxes). When the primary detector + secondary segmenter
    # both globally missed but the VLM already said the crop contains
    # the region of interest, run the OCR pipeline to *locate* the
    # text, then re-prompt the segmenter with a tight sub-crop around
    # that text region. The segmenter produces the final geometry; the
    # OCR region is only a hint.
    try:
        ocr_regions = await ocr_recognizer.detect_regions(task.crop_jpeg)
    except Exception as exc:
        logger.warning('text_hint_ocr_failed', crop_id=task.crop_id, error=str(exc))
        ocr_regions = []
    ocr_pick = ocr_recognizer.pick_best_plate_region(ocr_regions) if ocr_regions else None
    if ocr_pick is not None:
        task.detection_trace.append(f'{ocr_det_model}:text_hint:hit')
        sub_cand, _sub_box = await _resegment_from_text_hint(
            task.crop_jpeg, ocr_pick.bbox_norm, sam3
        )
        if sub_cand is None:
            task.detection_trace.append(f'{seg_name}:text_hint:miss')
        else:
            gate_ok, gate_reason = is_plausible_region_bbox(
                sub_cand.bbox_norm, task.vehicle_bbox_norm
            )
            if not gate_ok:
                task.detection_trace.append(f'{seg_name}:text_hint:hit')
                task.detection_trace.append(f'{seg_name}:text_hint:sanity_reject:{gate_reason}')
            else:
                plate_jpeg = _crop_region_jpeg(task.crop_jpeg, sub_cand.bbox_norm)
                outcome = await _verify_with_vlm(gemma, task.crop_id, plate_jpeg)
                ok, conf = outcome.ok, outcome.confidence
                if ok:
                    projected = crop_norm_to_source_norm(sub_cand.bbox_norm, task.vehicle_bbox_norm)
                    auto = await _auto_confirm_or_pending(
                        sam_score=sub_cand.score,
                        bbox_in_crop=sub_cand.bbox_norm,
                        vlm_high_conf=conf == 'high',
                    )
                    task.detection_trace.append(f'{seg_name}:text_hint:hit')
                    task.detection_trace.append(f'{seg_name}:text_hint:vlm_verify_ok')
                    # Pass OCR text through when the VLM read empty so
                    # the region text isn't lost.
                    text_out = outcome.text or ocr_pick.text
                    text_conf_out: str | None = outcome.text_confidence
                    if not outcome.text and ocr_pick.text:
                        rc = ocr_pick.rec_score
                        text_conf_out = 'high' if rc >= 0.8 else 'medium' if rc >= 0.5 else 'low'
                    task.update_doc = _region_write_doc(
                        plate_in_source=projected,
                        score=sub_cand.score,
                        detector=seg_name,
                        detector_version=seg_version,
                        chain=task.detection_trace,
                        plate_validated=auto or False,
                        plate_text=text_out,
                        plate_text_confidence=text_conf_out,
                    )
                    return
                task.detection_trace.append(f'{seg_name}:text_hint:hit')
                task.detection_trace.append(f'{seg_name}:text_hint:vlm_reject')
    elif ocr_regions:
        task.detection_trace.append(f'{ocr_det_model}:text_hint:no_region_shape')
    else:
        task.detection_trace.append(f'{ocr_det_model}:text_hint:miss')

    # ---- Step 5: All detectors missed. Queue for human review. ----
    _finalize_no_region(task)


# =============================================================================
# Bulk write
# =============================================================================

"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/sam_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.detection.cascade_detect import (
    DEFAULT_PROFILE,
    _now_iso,
    class_provenance,
    is_plausible_region_bbox,
    region_provenance,
)
from src.services.labeling.vlm_labeler import RegionCrop, VlmCombinedReply, VlmLabeler


logger = get_logger('curation_worker')


@dataclass
class _VerifyOutcome:
    """Compact record of one VLM verify-and-read call.

    Carried through the cascade so a single VLM call serves as both
    the is-this-a-real-region gate AND the OCR text source. Storing
    text on the same row that the bbox lands on means downstream
    training-set extraction (``mode=human_corrected`` etc.) can pull
    the text without a second query path.
    """

    ok: bool
    confidence: str  # 'high' | 'medium' | 'low'
    text: str | None
    text_confidence: str | None


async def _verify_with_vlm(vlm: VlmLabeler, crop_id: str, region_jpeg: bytes) -> _VerifyOutcome:
    """Verify and read a region in one VLM call.

    A ``confidence='low'`` ``is_plate=True`` verdict is treated as a
    rejection to keep the bar high — we'd rather route to the secondary
    segmenter than write a questionable region box. The returned
    outcome also carries ``text`` + ``text_confidence`` (None when the
    VLM couldn't read it or the verdict was rejected), which the caller
    threads into :func:`_region_write_doc`.
    """
    verdict = await vlm.verify_plate(RegionCrop(crop_id=crop_id, jpeg_bytes=region_jpeg))
    accepted = bool(verdict.is_plate) and verdict.confidence != 'low'
    return _VerifyOutcome(
        ok=accepted,
        confidence=verdict.confidence,
        text=verdict.text if accepted else None,
        text_confidence=verdict.text_confidence if accepted else None,
    )


# Region auto-confirm policy.
#
# By the time we reach the auto-confirm decision, the crop has passed two
# independent vision models:
#   1. A detector (primary detector or secondary segmenter) localized a
#      region-shaped area with score >= the detector's confidence floor.
#   2. The VLM looked at that region in isolation and said "yes, this is
#      a real region of interest" with at least medium confidence (low
#      is rejected upstream in ``_verify_with_vlm``).
#
# That's a 2-of-2 ensemble vote. The bbox-shape sanity check below exists
# only to catch detector hallucinations (e.g. a chrome bumper strip that
# happens to look region-like in isolation). The bounds (on
# ``DetectionProfile.auto_confirm_aspect`` / ``.auto_confirm_area_frac``)
# are intentionally loose to admit near-square regions, angled / partial
# regions, and small far-away regions.
_SKIP_VLM_VERIFY_SECONDARY_SCORE = float(os.environ.get('SAM3_SKIP_GEMMA_VERIFY_SCORE', '0.95'))
# Skip the VLM verify roundtrip when the secondary segmenter is very
# confident AND the bbox passes the same shape sanity check the VLM
# would do anyway. The VLM verify in this pipeline catches detector
# hallucinations (e.g. stickers, decals, bumper text). At score >= 0.95
# + region-shaped bbox the false-positive rate is low enough that the
# human review queue is the better backstop than a VLM roundtrip.
#
# This was chained: the VLM was the actual bottleneck — every secondary
# segmenter result queued for verification, capping the pipeline
# throughput. Skipping the verify on the strongest hits cuts the VLM
# load from this worker substantially and lets the shared VLM serve
# other queues (e.g. class labeling) instead.
#
# Override at runtime: SAM3_SKIP_GEMMA_VERIFY_SCORE=0.99 to be more
# conservative, or 0.90 for more aggressive skipping. Set to 1.01 to
# disable the skip entirely (everything still goes through the VLM).


def _bbox_shape_is_plausible(bbox_in_crop: tuple[float, float, float, float]) -> bool:
    """True if the crop-frame bbox is plausibly a region of interest.

    Thin wrapper over :func:`is_plausible_region_bbox` (Phase A3). The
    canonical helper carries the geometry rules; this wrapper preserves
    the boolean signature for internal call sites and adds the
    auto-confirm-specific minimum area floor
    (``DetectionProfile.auto_confirm_area_frac[0]``) which the canonical
    gate does not enforce.
    """
    ok, _reason = is_plausible_region_bbox(bbox_in_crop)
    if not ok:
        return False
    x1, y1, x2, y2 = bbox_in_crop
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area >= DEFAULT_PROFILE.auto_confirm_area_frac[0]


_VLM_TEXT_CONFIDENCE_MAP = {'high': 0.92, 'medium': 0.70, 'low': 0.40}


def _region_write_doc(
    *,
    plate_in_source: tuple[float, float, float, float],
    score: float,
    detector: str,
    detector_version: str,
    chain: list[str],
    plate_status: str = RegionStatus.DETECTED,
    plate_verified: bool = True,
    plate_validated: bool = False,
    verifier: str | None = 'gemma-4-e4b',
    verifier_version: str | None = '1',
    plate_text: str | None = None,
    plate_text_confidence: str | None = None,
    plate_text_source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the ``update_doc`` for a successful region-detection write.

    Centralizes the region-write shape so every cascade branch produces
    a consistent set of fields (incl. provenance + chain).
    """
    F = get_region_fields()
    doc: dict[str, Any] = {
        F.bbox_norm: list(plate_in_source),
        F.score: score,
        F.status: plate_status,
        F.verified: plate_verified,
        F.validated: plate_validated,
    }
    doc.update(
        region_provenance(
            detector=detector,
            detector_version=detector_version,
            bbox_frame='source',
            verifier=verifier if plate_verified else None,
            verifier_version=verifier_version if plate_verified else None,
        )
    )
    if chain:
        doc[F.detector_chain] = list(chain)
    if plate_text:
        doc[F.text] = plate_text
        doc[F.text_raw] = plate_text
        doc[F.text_source] = plate_text_source or 'gemma-4-e4b'
        doc[F.text_engine_version] = '1'
        if plate_text_confidence:
            doc[F.text_confidence] = _VLM_TEXT_CONFIDENCE_MAP.get(plate_text_confidence, 0.70)
    if extra:
        doc.update(extra)
    return doc


def _region_reject_doc(
    *,
    detector: str,
    detector_version: str,
    reason: str,
    chain: list[str],
) -> dict[str, Any]:
    """Compose the ``update_doc`` for a sanity-gate rejection.

    Records the detector identity so we can later quantify rejection
    rates per model, plus a ``RegionFields.rejection_reason`` keyword
    for triage.
    """
    F = get_region_fields()
    doc: dict[str, Any] = {
        F.status: RegionStatus.DETECTION_FAILED,
        F.rejection_reason: reason,
    }
    doc.update(
        region_provenance(
            detector=detector,
            detector_version=detector_version,
            bbox_frame='source',
        )
    )
    if chain:
        doc[F.detector_chain] = list(chain)
    return doc


def _combined_class_update(
    reply: VlmCombinedReply,
    class_names: list[str] | None,
    *,
    now: str | None = None,
    name_to_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the class-side update dict from a combined VLM reply.

    Always-applicable fields (make/model/plate_visible/gemma_verify_completed_at)
    are written regardless of whether a class was resolved. ``class_id`` /
    ``class_name`` only land when the reply contains a usable index into
    ``class_names``; otherwise the row is marked ``gemma_unmatched`` so the
    curator queue can grow the registry — same convention
    ``combined._try_combined_class_region`` uses.

    ``name_to_id`` is the registry's authoritative class_name -> class_id
    map. The reply's ``class_id`` is the *index* into ``class_names``
    (which is built filtering out deprecated entries), NOT a registry
    class_id. Looking the resolved name back up in the registry gives
    us the real class_id — without this remap, a stale index would land
    on the wrong class.

    Pass ``class_names=None`` (or an empty list) when the caller already
    has a trusted class label and the reply was generated with
    ``classify=False``; only the make/model/plate_visible fields are
    persisted in that case so the class column is preserved.
    """
    update: dict[str, Any] = {}
    ts = now or _now_iso()
    names = class_names or []
    if names and reply.class_id is not None and reply.class_id >= 0 and reply.class_id < len(names):
        cname = names[reply.class_id]
        # Resolve to the registry's authoritative class_id; fall back to
        # the index only if the name isn't in the registry (shouldn't
        # happen when names is built from reg.classes).
        cid = (name_to_id or {}).get(cname, int(reply.class_id))
        update.update(
            {
                'class_id': cid,
                'class_name': cname,
                'class_source': 'gemma',
                # Clearing label_source/class_validated: when this write
                # overwrites a prior class_source (e.g. a stale
                # 'v6_model'/'cluster_v6_majority_agreement' cohort), a
                # stale label_source='human'/class_validated=true would
                # otherwise persist and the doc would look like real
                # human ground truth even though the VLM now owns the
                # class.
                'label_source': 'gemma',
                'class_validated': False,
                'cluster_id': cid,
                'gemma_confidence': reply.class_confidence or 'low',
                'gemma_raw_label': cname,
                **class_provenance(
                    detector='gemma-4-e4b',
                    detector_version='1',
                    labeler='gemma-4-e4b',
                    labeled_at=ts,
                ),
            }
        )
    elif names:
        # We asked the VLM to classify and it returned -1 / null / out of range.
        update.update(
            {
                'class_source': 'gemma_unmatched',
                'label_source': 'gemma',
                'class_validated': False,
                'gemma_confidence': reply.class_confidence or 'low',
            }
        )
    # else: caller asked the VLM to SKIP classification — leave the
    # existing class fields untouched.
    if reply.make:
        update['gemma_vehicle_make'] = reply.make
    if reply.model:
        update['gemma_vehicle_model'] = reply.model
    update[get_region_fields().visible] = bool(reply.plate_visible)
    update['updated_at'] = ts
    # Marker: class + region resolved in one VLM call. Downstream
    # pipeline stages read this to skip a redundant class call.
    update['gemma_verify_completed_at'] = ts
    return update


def _combined_write_doc(
    *,
    reply: VlmCombinedReply,
    candidate_in_source: tuple[float, float, float, float],
    candidate_score: float,
    detector: str,
    detector_version: str,
    chain: list[str],
    class_names: list[str] | None,
    plate_validated: bool = False,
    name_to_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compose the happy-path ``detected`` write doc for a combined reply.

    Combines :func:`_region_write_doc` (region side) with
    :func:`_combined_class_update` (class side) so one helper produces
    every field the runner needs on a successful combined verification.
    ``name_to_id`` maps the resolved class_name back to the registry's
    authoritative class_id (see :func:`_combined_class_update`).
    """
    ts = _now_iso()
    region_doc = _region_write_doc(
        plate_in_source=candidate_in_source,
        score=candidate_score,
        detector=detector,
        detector_version=detector_version,
        chain=chain,
        plate_validated=plate_validated,
        plate_text=reply.plate_text,
        plate_text_confidence=reply.plate_confidence,
    )
    region_doc.update(_combined_class_update(reply, class_names, now=ts, name_to_id=name_to_id))
    return region_doc


async def _auto_confirm_or_pending(
    *,
    sam_score: float,
    bbox_in_crop: tuple[float, float, float, float],
    gemma_high_conf: bool,
) -> bool:
    """Decide whether the worker can auto-confirm without human review.

    Auto-confirm fires when the bbox shape is plausible AND either:
    * The VLM reports ``high`` confidence (strongest signal — when the
      VLM is certain we trust it even with a borderline detector
      score), OR
    * The detector score is >= ``_SKIP_VLM_VERIFY_SECONDARY_SCORE``
      (high-confidence detector + at least medium-confidence VLM is
      still a 2-of-2 vote).

    Anything else routes to the review tab where the operator confirms
    or tweaks the bbox.
    """
    # Two-signal validation — detector bbox + VLM 'high' verify is
    # sufficient regardless of in-crop bbox area (the VLM already saw
    # the region). The shape check is a sanity gate for the
    # lower-confidence fallbacks below.
    if gemma_high_conf:
        return True
    if not _bbox_shape_is_plausible(bbox_in_crop):
        return False
    return sam_score >= _SKIP_VLM_VERIFY_SECONDARY_SCORE

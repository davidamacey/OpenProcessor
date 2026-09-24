"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/sam_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from scripts.curation.worker.state import region_profile
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.vlm_class_attempt import (
    class_attempt_fields,
    empty_answer_reason_for_index,
)
from src.services.detection.cascade_detect import (
    _now_iso,
    class_provenance,
    is_plausible_region_bbox,
    region_provenance,
)
from src.services.detection.profile_registry import region_profile_or_neutral
from src.services.detection.region_text import TEXT_CHOICE_NONE, TEXT_CHOICE_VLM_ONLY
from src.services.detection.region_text_rules import region_text_rules
from src.services.labeling.vlm_client import DEFAULT_MODEL as VLM_MODEL_ID
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

    A ``confidence='low'`` ``is_region=True`` verdict is treated as a
    rejection to keep the bar high — we'd rather route to the secondary
    segmenter than write a questionable region box. The returned
    outcome also carries ``text`` + ``text_confidence`` (None when the
    VLM couldn't read it or the verdict was rejected), which the caller
    threads into :func:`_region_write_doc`.
    """
    verdict = await vlm.verify_plate(RegionCrop(crop_id=crop_id, jpeg_bytes=region_jpeg))
    accepted = bool(verdict.is_region) and verdict.confidence != 'low'
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
_SKIP_VLM_VERIFY_SECONDARY_SCORE = float(
    os.environ.get('SAM3_SKIP_VLM_VERIFY_SCORE')
    or os.environ.get('SAM3_SKIP_GEMMA_VERIFY_SCORE')
    or '0.95'
)
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
# Override at runtime: SAM3_SKIP_VLM_VERIFY_SCORE=0.99 to be more
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
    return area >= region_profile().auto_confirm_area_frac[0]


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
    auto_confirmed: bool = False,
    verifier: str | None = VLM_MODEL_ID,
    verifier_version: str | None = '1',
    plate_text: str | None = None,
    plate_text_confidence: str | None = None,
    plate_text_source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the ``update_doc`` for a successful region-detection write.

    Centralizes the region-write shape so every cascade branch produces
    a consistent set of fields (incl. provenance + chain). The worker
    never validates a region -- ``RegionFields.validated`` is human-only;
    its auto-confirm policy's verdict is ``RegionFields.auto_confirmed``.
    """
    F = get_region_fields()
    doc: dict[str, Any] = {
        F.bbox_norm: list(plate_in_source),
        F.score: score,
        F.status: plate_status,
        F.verified: plate_verified,
        F.validated: False,
        F.auto_confirmed: auto_confirmed,
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
    # An accepted box supersedes any candidate an earlier pass rejected.
    doc.update(dict.fromkeys(candidate_fields(F)))
    doc[F.rejection_reason] = None
    invalid = (
        region_text_rules(region_profile_or_neutral()).invalid_reason(plate_text)
        if plate_text
        else None
    )
    if plate_text and invalid:
        # Not text (a prompt placeholder, a "can't read it" answer, ...):
        # keep the reading for audit, write no region text.
        doc[F.text_vlm] = plate_text
        doc[F.text_vlm_invalid] = invalid
        doc[F.text_choice] = TEXT_CHOICE_NONE
    elif plate_text:
        doc[F.text] = plate_text
        doc[F.text_raw] = plate_text
        doc[F.text_source] = plate_text_source or VLM_MODEL_ID
        doc[F.text_engine_version] = '1'
        doc[F.text_choice] = TEXT_CHOICE_VLM_ONLY
        if plate_text_confidence:
            doc[F.text_confidence] = _VLM_TEXT_CONFIDENCE_MAP.get(plate_text_confidence, 0.70)
    if extra:
        doc.update(extra)
    return doc


def candidate_fields(F: Any) -> tuple[str, ...]:
    """Storage names of the rejected-candidate fields."""
    return (
        F.candidate_bbox_norm,
        F.candidate_score,
        F.candidate_detector,
        F.candidate_detector_version,
        F.candidate_source,
    )


def candidate_reject_doc(
    *,
    candidate_in_source: tuple[float, float, float, float] | None,
    candidate_score: float,
    detector: str,
    detector_version: str,
    candidate_source: str,
    reason: str,
    chain: list[str],
    bbox_correct: bool | None = None,
) -> dict[str, Any]:
    """Region side of a ``verify_rejected`` write.

    The rejected box is kept in the ``candidate_*`` fields -- never in
    ``bbox_norm``, which every reader treats as an accepted region -- with
    its detector and score, plus the rejection reason and the verifier's
    box verdict, so a human can review the rejection and reverse it
    (confirming promotes the candidate). Any accepted box a
    pending-verification item carried is cleared: the verifier rejected it.
    """
    F = get_region_fields()
    doc: dict[str, Any] = {
        F.status: RegionStatus.VERIFY_REJECTED,
        F.rejection_reason: reason,
        F.bbox_norm: None,
        F.score: None,
        F.detector_chain: list(chain),
    }
    if bbox_correct is not None:
        doc[F.bbox_correct] = bbox_correct
    if candidate_in_source is not None:
        doc.update(
            {
                F.candidate_bbox_norm: list(candidate_in_source),
                F.candidate_score: candidate_score,
                F.candidate_detector: detector,
                F.candidate_detector_version: detector_version,
                F.candidate_source: candidate_source,
            }
        )
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

    Always-applicable fields (make/model/plate_visible/vlm_verify_completed_at)
    are written regardless of whether a class was resolved. ``class_id`` /
    ``class_name`` only land when the reply contains a usable index into
    ``class_names``. A reply that *named* a label outside the catalog
    (``reply.class_raw``) marks the row ``vlm_unmatched`` with that label
    in ``vlm_raw_class`` so the curator queue can grow the registry. A
    reply with no class at all (``null`` / ``-1`` / out-of-range index)
    leaves every class field untouched and only records the attempt
    (:mod:`src.services.curation.vlm_class_attempt`).

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
                'class_source': 'vlm',
                # Clearing label_source/class_validated: when this write
                # overwrites a prior class_source (e.g. a stale
                # 'item_model'/'cluster_majority_agreement' cohort), a
                # stale label_source='human'/class_validated=true would
                # otherwise persist and the doc would look like real
                # human ground truth even though the VLM now owns the
                # class.
                'label_source': 'vlm',
                'class_validated': False,
                'cluster_id': cid,
                'vlm_confidence': reply.class_confidence or 'low',
                'vlm_raw_label': cname,
                **class_provenance(
                    detector=VLM_MODEL_ID,
                    detector_version='1',
                    labeler=VLM_MODEL_ID,
                    labeled_at=ts,
                ),
            }
        )
        update.update(class_attempt_fields(ts))
    elif names and reply.class_raw:
        update.update(
            {
                'class_source': 'vlm_unmatched',
                'label_source': 'vlm',
                'class_validated': False,
                'vlm_confidence': reply.class_confidence or 'low',
                'vlm_raw_class': reply.class_raw,
                'vlm_raw_label': reply.class_raw,
                **class_attempt_fields(ts),
            }
        )
    elif names:
        update.update(
            class_attempt_fields(ts, empty_answer_reason_for_index(reply.class_id, len(names)))
        )
    # else: caller asked the VLM to SKIP classification — leave the
    # existing class fields untouched.
    if reply.make:
        update['vlm_item_make'] = reply.make
    if reply.model:
        update['vlm_item_model'] = reply.model
    update[get_region_fields().visible] = bool(reply.plate_visible)
    update['updated_at'] = ts
    # Marker: class + region resolved in one VLM call. Downstream
    # pipeline stages read this to skip a redundant class call.
    update['vlm_verify_completed_at'] = ts
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
    auto_confirmed: bool = False,
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
        auto_confirmed=auto_confirmed,
        plate_text=reply.plate_text,
        plate_text_confidence=reply.plate_confidence,
    )
    region_doc.update(_combined_class_update(reply, class_names, now=ts, name_to_id=name_to_id))
    return region_doc


async def _auto_confirm_or_pending(
    *,
    sam_score: float,
    bbox_in_crop: tuple[float, float, float, float],
    vlm_high_conf: bool,
) -> bool:
    """Decide whether the worker's auto-confirm policy accepts the box.

    The verdict is recorded as ``RegionFields.auto_confirmed`` -- never as
    human validation -- so an auto-confirmed region is accepted
    (``detected``) yet still reviewable in the human region queue.

    Auto-confirm fires when the bbox shape is plausible AND either:
    * The VLM reports ``high`` confidence (strongest signal — when the
      VLM is certain we trust it even with a borderline detector
      score), OR
    * The detector score is >= ``_SKIP_VLM_VERIFY_SECONDARY_SCORE``
      (high-confidence detector + at least medium-confidence VLM is
      still a 2-of-2 vote).

    Anything else is accepted unconfirmed; either way the operator can
    confirm or tweak the bbox from the review tab.
    """
    # Two-signal validation — detector bbox + VLM 'high' verify is
    # sufficient regardless of in-crop bbox area (the VLM already saw
    # the region). The shape check is a sanity gate for the
    # lower-confidence fallbacks below.
    if vlm_high_conf:
        return True
    if not _bbox_shape_is_plausible(bbox_in_crop):
        return False
    return sam_score >= _SKIP_VLM_VERIFY_SECONDARY_SCORE

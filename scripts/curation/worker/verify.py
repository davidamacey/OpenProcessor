"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/region_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scripts.curation.worker.state import _ItemTask, region_profile
from src.config import get_region_fields
from src.config.region_rejection import (
    REJECT_REASON_NO_VERDICT,
    REJECT_REASON_SANITY_PREFIX,
    REJECT_REASON_VERIFIER,
)
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.region_boxes import RegionBox, derive_status, new_box_placeholder
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


if TYPE_CHECKING:
    from src.services.labeling.region_overlay import VlmBoxVerdict


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


async def _verify_with_vlm(
    vlm: VlmLabeler, task: _ItemTask, region_jpeg: bytes
) -> _VerifyOutcome | None:
    """Verify and read a region in one VLM call.

    A ``confidence='low'`` ``is_region=True`` verdict is treated as a
    rejection to keep the bar high — we'd rather route to the secondary
    segmenter than write a questionable region box. The returned
    outcome also carries ``text`` + ``text_confidence`` (None when the
    VLM couldn't read it or the verdict was rejected), which the caller
    threads into :func:`_region_write_doc`.

    Returns ``None`` when the VLM answered with no usable verdict (see
    :py:meth:`VlmLabeler.verify_region`) so the cascade can leave the
    crop pending for a retry instead of treating "no answer" as a
    rejection and falling through to the next detector. Raises
    :class:`VlmTransportError` when the call itself failed, so an outage
    is never counted as a no-verdict reply.

    Sets ``task.vlm_called`` (minor 5, W2 review): the round trip
    happened, whatever the verdict, so any write this task ends up
    producing this pass may be stamped ``vlm_prompt_pack``.
    """
    verdict = await vlm.verify_region(
        RegionCrop(crop_id=task.crop_id, jpeg_bytes=region_jpeg), raise_on_transport=True
    )
    task.vlm_called = True
    if verdict is None:
        return None
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
_SKIP_VLM_VERIFY_SECONDARY_SCORE = float(os.environ.get('OP_SEGMENTER_SKIP_VERIFY_SCORE') or '0.95')
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
# Override at runtime: OP_SEGMENTER_SKIP_VERIFY_SCORE=0.99 to be more
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
    region_in_source: tuple[float, float, float, float],
    score: float,
    detector: str,
    detector_version: str,
    chain: list[str],
    region_status: str = RegionStatus.DETECTED,
    region_verified: bool = True,
    auto_confirmed: bool = False,
    verifier: str | None = VLM_MODEL_ID,
    verifier_version: str | None = '1',
    region_text_reply: str | None = None,
    region_text_confidence: str | None = None,
    region_text_source: str | None = None,
    confidence: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the ``update_doc`` for a successful region-detection write.

    Centralizes the region-write shape so every cascade branch produces
    a consistent set of fields (incl. provenance + chain). The worker
    never validates a region -- ``RegionFields.validated`` is human-only;
    its auto-confirm policy's verdict is ``RegionFields.auto_confirmed``.

    ``confidence`` is the VLM verifier's box-confidence verdict
    (``high``/``medium``/``low``, ``RegionFields.confidence`` ==
    ``reply.region_confidence`` / ``outcome.confidence``) -- distinct from
    ``region_text_confidence``, which grades the *text reading*.
    """
    F = get_region_fields()
    doc: dict[str, Any] = {
        F.bbox_norm: list(region_in_source),
        F.score: score,
        F.status: region_status,
        F.verified: region_verified,
        F.validated: False,
        F.auto_confirmed: auto_confirmed,
    }
    if confidence:
        doc[F.confidence] = confidence
    doc.update(
        region_provenance(
            detector=detector,
            detector_version=detector_version,
            bbox_frame='source',
            verifier=verifier if region_verified else None,
            verifier_version=verifier_version if region_verified else None,
        )
    )
    if chain:
        doc[F.detector_chain] = list(chain)
    # An accepted box supersedes any candidate an earlier pass rejected.
    doc.update(dict.fromkeys(candidate_fields(F)))
    doc[F.rejection_reason] = None
    profile = region_profile_or_neutral()
    if region_text_reply and not profile.reads_text:
        logger.debug('region_text_ignored_text_free', profile=profile.name)
        region_text_reply = None
    invalid = (
        region_text_rules(profile).invalid_reason(region_text_reply) if region_text_reply else None
    )
    if region_text_reply and invalid:
        # Not text (a prompt placeholder, a "can't read it" answer, ...):
        # keep the reading for audit, write no region text.
        doc[F.text_vlm] = region_text_reply
        doc[F.text_vlm_invalid] = invalid
        doc[F.text_choice] = TEXT_CHOICE_NONE
    elif region_text_reply:
        doc[F.text] = region_text_reply
        doc[F.text_raw] = region_text_reply
        doc[F.text_source] = region_text_source or VLM_MODEL_ID
        doc[F.text_engine_version] = '1'
        doc[F.text_choice] = TEXT_CHOICE_VLM_ONLY
        if region_text_confidence:
            doc[F.text_confidence] = _VLM_TEXT_CONFIDENCE_MAP.get(region_text_confidence, 0.70)
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

    Always-applicable fields (make/model/region_visible/vlm_verify_completed_at)
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
    ``classify=False``; only the make/model/region_visible fields are
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
    update[get_region_fields().visible] = bool(reply.region_visible)
    update['updated_at'] = ts
    # Marker: class + region resolved in one VLM call. Downstream
    # pipeline stages read this to skip a redundant class call.
    update['vlm_verify_completed_at'] = ts
    return update


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


# =============================================================================
# W8.5: verdict -> box-list storage (verdicts_to_boxes)
# =============================================================================
#
# Wired into the streaming runner's stage_b_combined (runner.py): the
# live pipeline calls select_region_candidates() to build the candidate
# list, then this module's verdicts_to_boxes() to resolve the VLM's
# per-box verdicts into RegionBox entries, written via
# region_boxes.boxes_write_fields(). candidate_reject_doc() /
# no_verdict_reject_doc() (no_verdict.py) are the pre-W8 single-candidate
# write builders -- no longer called from runner.py, left defined
# (candidate_reject_doc still backs no_verdict.py's own helpers) rather
# than swept this pass; see the W8 handback report.


@dataclass(frozen=True)
class TaskBoxInput:
    """One candidate box going into a verify call (a bounded stand-in for
    the eventual worker ``TaskBox`` -- see W8.5). ``box_id`` is set for a
    box read back from storage (``pending_verification``); ``None`` for a
    fresh candidate, which gets the next id in :func:`verdicts_to_boxes`.
    """

    bbox_in_crop: tuple[float, float, float, float]
    bbox_in_source: tuple[float, float, float, float]
    score: float
    detector: str
    detector_version: str
    source: str
    box_id: str | None = None
    hint_text: str | None = None
    hint_text_confidence: float | None = None


def verdicts_to_boxes(
    candidates: list[TaskBoxInput],
    verdicts: list[VlmBoxVerdict],
    *,
    item_bbox_norm: tuple[float, float, float, float] | None = None,
    now: str | None = None,
    force_resolve: bool = False,
) -> tuple[list[RegionBox], RegionStatus | None, dict[str, Any]]:
    """Map a combined VLM reply's per-box verdicts onto stored boxes (W8.5).

    ``candidates`` and ``verdicts`` are aligned by position (both length
    N; ``verdicts[i].box == i + 1``). Every entry in the returned list
    carries its own ``box_id`` (Cropwright C3/Q15) -- a candidate read
    back from storage (``cand.box_id`` set, W8 B1) keeps that real id;
    a fresh candidate (``cand.box_id is None``) gets a PLACEHOLDER id
    (:func:`~src.services.curation.region_boxes.new_box_placeholder`),
    not a real ``b<N>`` one. W8 M1: a real id is only safe to mint
    against the CURRENT ``region_box_seq`` high-water mark, which this
    function -- called at task-processing time, potentially long before
    the write actually lands -- does not have; the caller's write path
    (``bulk_writer._merge``) finalizes placeholders into real ids
    immediately before the write, against the live doc.

    Returns ``(boxes, status, extra)``:

    - Every verdict ``bbox_correct is None`` (no box got a verdict at
      all) and ``force_resolve`` is False: ``([], None, {'no_verdict':
      True})`` -- the caller retries (the no-verdict cap), same as
      today's single-box no-verdict path. ``force_resolve=True`` (the cap
      reached) resolves every no-verdict box as ``rejected`` /
      ``REJECT_REASON_NO_VERDICT`` instead of signalling a retry.
    - Otherwise: one :class:`RegionBox` per candidate --
      ``bbox_correct=True`` and the box passes the sanity gate ->
      ``accepted``; ``True`` but sanity fails -> ``rejected`` /
      ``sanity_reject:<reason>``; ``False`` -> ``rejected`` /
      ``region_visible_elsewhere``; ``None`` -> ``rejected`` /
      ``verifier_no_verdict`` (this box specifically got no verdict, but
      at least one sibling did). ``status = derive_status(boxes,
      empty_status=RegionStatus.NO_REGION_BOX)``.
    """
    now = now or _now_iso()
    any_verdict = any(v.bbox_correct is not None for v in verdicts)
    if not any_verdict and not force_resolve:
        return [], None, {'no_verdict': True}

    boxes: list[RegionBox] = []
    for i, (cand, verdict) in enumerate(zip(candidates, verdicts, strict=True)):
        box_id = cand.box_id if cand.box_id is not None else new_box_placeholder(i)
        state: str
        rejection_reason: str | None = None
        bbox_correct = verdict.bbox_correct
        if bbox_correct is True:
            gate_ok, gate_reason = is_plausible_region_bbox(cand.bbox_in_crop, item_bbox_norm)
            if gate_ok:
                state = 'accepted'
            else:
                state = 'rejected'
                rejection_reason = f'{REJECT_REASON_SANITY_PREFIX}{gate_reason}'
        elif bbox_correct is False:
            state = 'rejected'
            rejection_reason = REJECT_REASON_VERIFIER
        else:
            state = 'rejected'
            rejection_reason = REJECT_REASON_NO_VERDICT
        boxes.append(
            RegionBox(
                box_id=box_id,
                bbox_norm=cand.bbox_in_source,
                state=state,
                score=cand.score,
                detector=cand.detector,
                detector_version=cand.detector_version,
                source=cand.source,
                bbox_correct=bbox_correct,
                confidence=verdict.confidence,
                rejection_reason=rejection_reason,
                text=verdict.text_reply or cand.hint_text,
                detected_at=now,
            )
        )

    status = derive_status(boxes, empty_status=RegionStatus.NO_REGION_BOX)
    return boxes, status, {}


# =============================================================================
# W8 M2 fix (pipeline-wiring review, 2026-09-27): item-level verification
# fields for a box-list write.
# =============================================================================
#
# Pre-W8, every combined-verify write set RegionFields.verified/verifier/
# verifier_version/verified_at/validated/auto_confirmed at the item level
# (_region_write_doc above). The W8 box-list rewrite dropped these
# entirely -- readers that still filter/sort on them (regions.py's
# `verified` filter, the detector_blind_spots/low_conf_correct training
# cohorts, region_requeue.py, the auto-confirm review semantics) went
# blind to every fresh worker write. Box-aware definition: `verified`
# means the VLM actually rendered a verdict this write (`reply is not
# None`); `auto_confirmed` means at least one box was accepted AND every
# accepted box independently passes `_auto_confirm_or_pending`.


def item_verification_fields(
    *,
    verified: bool,
    auto_confirmed: bool = False,
    now: str | None = None,
) -> dict[str, Any]:
    """Item-level ``verified``/``verifier*``/``validated``/``auto_confirmed``
    fields for a box-list write (W8 M2).

    ``validated`` is always ``False`` here -- the worker never validates a
    region, only a human does (``RegionFields.validated`` semantics,
    unchanged from pre-W8). ``verifier``/``verifier_version`` are cleared
    (``None``) when ``verified`` is False, mirroring the pre-W8
    ``_region_write_doc`` behaviour.
    """
    F = get_region_fields()
    doc: dict[str, Any] = {
        F.validated: False,
        F.auto_confirmed: auto_confirmed,
        F.verified: verified,
        F.verifier: VLM_MODEL_ID if verified else None,
        F.verifier_version: '1' if verified else None,
    }
    if verified:
        doc[F.verified_at] = now or _now_iso()
    return doc


async def boxes_auto_confirmed(boxes: list[RegionBox], candidates: list[TaskBoxInput]) -> bool:
    """W8 M2: the box-aware ``auto_confirmed`` rule.

    At least one accepted box, AND every accepted box independently
    passes :func:`_auto_confirm_or_pending` (aligned by position with
    ``candidates`` -- the same alignment ``verdicts_to_boxes`` produces).
    A set with zero accepted boxes is never auto-confirmed.
    """
    accepted_pairs = [
        (b, c) for b, c in zip(boxes, candidates, strict=True) if b.state == 'accepted'
    ]
    if not accepted_pairs:
        return False
    for box, cand in accepted_pairs:
        ok = await _auto_confirm_or_pending(
            sam_score=cand.score,
            bbox_in_crop=cand.bbox_in_crop,
            vlm_high_conf=box.confidence == 'high',
        )
        if not ok:
            return False
    return True

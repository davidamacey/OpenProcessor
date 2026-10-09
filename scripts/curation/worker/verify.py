"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/region_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scripts.curation.worker.cascade import _source_to_crop
from scripts.curation.worker.state import region_profile
from src.config import get_region_fields
from src.config.region_rejection import (
    REJECT_REASON_NO_VERDICT,
    REJECT_REASON_SANITY_PREFIX,
    REJECT_REASON_VERIFIER,
)
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.region_boxes import (
    RegionBox,
    derive_status,
    has_human_text,
    is_human_owned,
    new_box_placeholder,
)
from src.services.curation.vlm_class_attempt import (
    class_attempt_fields,
    empty_answer_reason_for_index,
)
from src.services.detection.cascade_detect import (
    _now_iso,
    class_provenance,
    is_plausible_region_bbox,
)


if TYPE_CHECKING:
    from src.config import DetectionProfile
    from src.services.labeling.region_overlay import VlmBoxVerdict
    from src.services.labeling.vlm_labeler import VlmCombinedReply


logger = get_logger('curation_worker')


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


def _bbox_shape_is_plausible(
    bbox_in_crop: tuple[float, float, float, float], profile: DetectionProfile | None = None
) -> bool:
    """True if the crop-frame bbox is plausibly a region of interest.

    Thin wrapper over :func:`is_plausible_region_bbox` (Phase A3). The
    canonical helper carries the geometry rules; this wrapper preserves
    the boolean signature for internal call sites and adds the
    auto-confirm-specific minimum area floor
    (``DetectionProfile.auto_confirm_area_frac[0]``) which the canonical
    gate does not enforce. ``profile`` defaults to the active profile; a
    caller running someone else's profile (a test run of a draft) passes it.
    """
    ok, _reason = is_plausible_region_bbox(bbox_in_crop)
    if not ok:
        return False
    x1, y1, x2, y2 = bbox_in_crop
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area >= (profile or region_profile()).auto_confirm_area_frac[0]


def _combined_class_update(
    reply: VlmCombinedReply,
    class_names: list[str] | None,
    *,
    vlm_model: str | None,
    now: str | None = None,
    name_to_id: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the class-side update dict from a combined VLM reply.

    ``vlm_model`` is the resolved model of the endpoint that produced
    ``reply`` (the runtime's ``vlm_identity.model``): it is the class
    ``detector`` / ``labeler`` provenance, so a hot switch never leaves a
    stale process-wide model id on a write.

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
        if vlm_model is None:
            msg = 'a class was resolved from a VLM reply but the answering model is unknown'
            raise ValueError(msg)
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
                    detector=vlm_model,
                    detector_version='1',
                    labeler=vlm_model,
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
    profile: DetectionProfile | None = None,
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
    if not _bbox_shape_is_plausible(bbox_in_crop, profile):
        return False
    return sam_score >= _SKIP_VLM_VERIFY_SECONDARY_SCORE


# =============================================================================
# W8.5: verdict -> box-list storage (verdicts_to_boxes)
# =============================================================================
#
# Wired into the streaming runner's stage_b_combined (stage_b.py): the
# live pipeline calls select_region_candidates() to build the candidate
# list, then this module's verdicts_to_boxes() to resolve the VLM's
# per-box verdicts into RegionBox entries, written via
# region_boxes.boxes_write_fields().


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
    #: The stored box this candidate re-verifies (``None`` for a fresh
    #: detection). Everything the pass does not judge (cluster placement,
    #: human text, detector provenance) is kept from it.
    stored: RegionBox | None = None


def task_box_from_stored(
    box: RegionBox, *, item_bbox_norm: tuple[float, float, float, float]
) -> TaskBoxInput:
    """W8 B1 fix: wrap one stored ``proposed`` box as a VLM re-verify
    candidate, preserving its ``box_id`` (never minting a fresh one --
    this is the same box, going back through verification, not a new
    detection) plus its stored score/detector/source. The real source of
    truth for Path 1 (``pending_verification``), never the legacy
    single-scalar fields.
    """
    return TaskBoxInput(
        bbox_in_crop=_source_to_crop(box.bbox_norm, item_bbox_norm),
        bbox_in_source=box.bbox_norm,
        score=box.score or 0.0,
        detector=box.detector or 'human',
        detector_version=box.detector_version or '1',
        source=box.source or 'human',
        box_id=box.box_id,
        stored=box,
    )


def candidate_actor(cand: TaskBoxInput) -> str | None:
    """The detector a candidate's detector-chain entries are filed under, or
    ``None`` for a stored box a human owns or that has no detector: a person
    is not a detector, and the chain feeds the training-cohort queries."""
    if cand.stored is not None and (is_human_owned(cand.stored) or not cand.stored.detector):
        return None
    return cand.detector


def chain_entry(actor: str | None, event: str) -> list[str]:
    """``[f'{actor}:{event}']``, or nothing for a non-detector actor."""
    return [] if actor is None else [f'{actor}:{event}']


def candidate_box(
    cand: TaskBoxInput, *, fallback_id: str, now: str | None = None, **updates: Any
) -> RegionBox:
    """The box a verdict on ``cand`` produces, with ``updates`` applied: the
    stored box with only those fields changed when ``cand`` re-verifies one
    (its id, cluster placement, detector provenance and human text survive),
    otherwise a fresh box under ``fallback_id`` (a placeholder until the
    writer mints a real id)."""
    if cand.stored is not None:
        return dataclasses.replace(cand.stored, **updates)
    return RegionBox(
        box_id=cand.box_id if cand.box_id is not None else fallback_id,
        bbox_norm=cand.bbox_in_source,
        score=cand.score,
        detector=cand.detector,
        detector_version=cand.detector_version,
        source=cand.source,
        detected_at=now,
        **updates,
    )


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
        text_update = (
            {} if has_human_text(cand.stored) else {'text': verdict.text_reply or cand.hint_text}
        )
        boxes.append(
            candidate_box(
                cand,
                fallback_id=new_box_placeholder(i),
                now=now,
                state=state,
                bbox_correct=bbox_correct,
                confidence=verdict.confidence,
                rejection_reason=rejection_reason,
                **text_update,
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
# blind to every fresh worker write. Box-aware definition (corrected by
# the 2026-09-27 re-review, R-M4: the first cut used `reply is not None`,
# which is also true for an all-rejected verdict set and contradicted the
# existing invariant in test_region_status_invariants.py plus the pre-W8
# write, which only ever set `verified=True` on an ACCEPTED write):
# `verified` means the VLM actually CONFIRMED a region this write -- at
# least one box in the final set is `accepted`; `auto_confirmed` means at
# least one box was accepted AND every accepted box independently passes
# `_auto_confirm_or_pending`.


def item_verification_fields(
    *,
    verified: bool,
    verifier: str | None,
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
        F.verifier: verifier if verified else None,
        F.verifier_version: '1' if verified else None,
    }
    if verified:
        doc[F.verified_at] = now or _now_iso()
    return doc


async def boxes_auto_confirmed(
    boxes: list[RegionBox],
    candidates: list[TaskBoxInput],
    profile: DetectionProfile | None = None,
) -> bool:
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
            profile=profile,
        )
        if not ok:
            return False
    return True

"""One combined VLM reply -> the boxes, status and write fields it resolves to.

Extracted from the worker's combined stage (``runner.stage_b_combined``) so
the live pipeline and the test-on-crop preview (``POST /prompt_packs/test``,
``POST /region_profiles/test``) resolve a reply through the same code: the
``region_visible=False`` branch, the no-verdict outcome, per-box verdicts
-> accepted/rejected boxes, per-box text, auto-confirm, the class update
and the item-level verification fields.

Pure with respect to the task and the store: the caller owns the retry
bookkeeping (no-verdict counter, metrics) and the write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scripts.curation.worker.region_text_stage import (
    _box_with_resolved_text,
    apply_region_text,
    apply_text_hint_fallback,
    resolve_rejected_box_text,
)
from scripts.curation.worker.verify import (
    _combined_class_update,
    boxes_auto_confirmed,
    candidate_actor,
    candidate_box,
    chain_entry,
    item_verification_fields,
    verdicts_to_boxes,
)
from src.config.region_rejection import REJECT_REASON_VERIFIER
from src.config.region_state import RegionStatus
from src.services.curation.ingest_class_sources import (
    CLUSTER_MAJORITY_CLASS_SOURCE,
    classifier_class_sources,
)
from src.services.curation.region_boxes import derive_status, has_human_text, new_box_placeholder
from src.services.labeling.region_overlay import VlmBoxVerdict


if TYPE_CHECKING:
    from scripts.curation.worker.verify import TaskBoxInput
    from src.config import DetectionProfile
    from src.services.curation.region_boxes import RegionBox
    from src.services.detection.cascade_detect import PaddleOcrTextRecognizer
    from src.services.detection.region_text_rules import RegionTextRules
    from src.services.labeling.vlm_models import VlmCombinedReply


# A crop whose class already came from a classifier or a cluster at or above
# this confidence is not re-classified by the combined call (the caller
# already has a trusted class). Same threshold as the legacy cascade's
# combined-cohort gate.
CLASSIFIER_HIGH_CONF_THRESHOLD = 0.80


def should_classify(
    *,
    class_validated: bool,
    stored_class_source: str,
    test_holdout: bool,
    class_confidence: float,
    registry_loaded: bool,
) -> bool:
    """Whether the combined prompt asks the VLM for the item class.

    False when the registry did not load, the class was confirmed by a
    human (a human verdict is never silently reclassified), the crop is a
    frozen test-holdout crop (class-field guard only; region fields stay
    unconditional), or it already carries a high-confidence classifier or
    cluster-majority class.
    """
    if not registry_loaded:
        return False
    if class_validated or stored_class_source.startswith('human'):
        return False
    if test_holdout:
        return False
    return not (
        stored_class_source in (classifier_class_sources() | {CLUSTER_MAJORITY_CLASS_SOURCE})
        and class_confidence >= CLASSIFIER_HIGH_CONF_THRESHOLD
    )


@dataclass(frozen=True)
class CombinedResolution:
    #: ``'no_verdict'``: no candidate got any verdict (nothing to write; the
    #: caller retries up to its cap, then resolves with ``force_resolve``).
    outcome: str
    boxes: list[RegionBox] = field(default_factory=list)
    status: RegionStatus | None = None
    #: Class update + item-level verification fields for the write.
    extra: dict[str, Any] = field(default_factory=dict)
    #: Detector-chain entries this resolution adds (after the ``:hit`` one).
    trace: list[str] = field(default_factory=list)
    no_region_visible: bool = False
    #: At least one candidate was rejected by the verifier (metrics only).
    bbox_wrong: bool = False


async def resolve_combined_reply(
    candidates: list[TaskBoxInput],
    reply: VlmCombinedReply | None,
    *,
    item_bbox_norm: tuple[float, float, float, float],
    reverify: bool,
    effective_class_names: list[str] | None,
    name_to_id: dict[str, int],
    vlm_model: str | None,
    profile: DetectionProfile,
    rules: RegionTextRules,
    ocr: PaddleOcrTextRecognizer | None,
    crop_jpeg: bytes | None,
    crop_id: str,
    force_resolve: bool = False,
) -> CombinedResolution:
    """Resolve ``reply`` (``None``: no parseable entry) for one item.

    ``reverify`` is the item's stored ``proposed`` box(es) going back
    through verification (the ``region_visible=False`` answer then rejects
    them, keeping their ids, instead of writing an empty terminal status).
    ``force_resolve`` is the no-verdict cap: every box resolves ``rejected``.
    """
    actor = candidate_actor(candidates[0]) if candidates else 'unknown'
    class_update = (
        None
        if reply is None
        else _combined_class_update(
            reply, effective_class_names, name_to_id=name_to_id, vlm_model=vlm_model
        )
    )

    if reply is not None and not reply.region_visible:
        # No region in this crop, regardless of any per-box verdict.
        if reverify:
            # Stored `proposed` boxes going back through verification: writing
            # an empty list would leave them `proposed` forever (every poll
            # another VLM call). Reject each, keeping its id, so a human can
            # still reverse the verdict; the status then comes from the FULL
            # list like every other verdict branch.
            boxes = [
                resolve_rejected_box_text(
                    candidate_box(
                        cand,
                        fallback_id=new_box_placeholder(i),
                        state='rejected',
                        rejection_reason=REJECT_REASON_VERIFIER,
                    ),
                    profile=profile,
                    rules=rules,
                )
                for i, cand in enumerate(candidates)
            ]
            status = derive_status(boxes, empty_status=RegionStatus.NO_REGION_VISIBLE)
        else:
            boxes = []
            status = RegionStatus.NO_REGION_VISIBLE
        return CombinedResolution(
            outcome='resolved',
            boxes=boxes,
            status=status,
            extra={
                **(class_update or {}),
                # `verified` means the VLM CONFIRMED a region: never here.
                **item_verification_fields(verified=False, verifier=None),
            },
            trace=chain_entry(actor, 'combined_no_region_visible'),
            no_region_visible=True,
        )

    no_verdicts = [
        VlmBoxVerdict(box=i, bbox_correct=None, confidence=None)
        for i in range(1, len(candidates) + 1)
    ]
    boxes, verdict_status, extra = verdicts_to_boxes(
        candidates,
        reply.region_boxes if reply is not None else no_verdicts,
        item_bbox_norm=item_bbox_norm,
        force_resolve=force_resolve,
    )
    if extra.get('no_verdict'):
        return CombinedResolution(outcome='no_verdict')
    assert verdict_status is not None  # only the no-verdict sentinel returns None

    if any(b.state == 'accepted' for b in boxes):
        trace = chain_entry(actor, 'combined_verify_ok')
    elif len(boxes) == 1 and boxes[0].rejection_reason:
        trace = chain_entry(actor, f'combined_verify_reject:{boxes[0].rejection_reason}')
    else:
        trace = chain_entry(actor, 'combined_verify_reject')

    # Per-box text (VLM + OCR fallback) for every accepted box, using that
    # box's own crop-frame bbox. A non-accepted box never carries a raw,
    # unvalidated VLM text reply (W8 M6).
    resolved: list[RegionBox] = []
    for box, cand in zip(boxes, candidates, strict=True):
        if box.state != 'accepted':
            resolved.append(resolve_rejected_box_text(box, profile=profile, rules=rules))
            continue
        if has_human_text(box):
            resolved.append(box)
            continue
        text_doc: dict[str, Any] = {}
        await apply_region_text(
            text_doc,
            ocr=ocr,  # type: ignore[arg-type]  # only read when the profile reads text
            crop_jpeg=crop_jpeg,
            region_in_crop=cand.bbox_in_crop,
            profile=profile,
            crop_id=crop_id,
            vlm_text=box.text,
            vlm_confidence=box.confidence,
            vlm_available=True,
            vlm_model=vlm_model,
            rules=rules,
        )
        # The VLM read nothing but the item-crop OCR hit that seeded this box
        # did (text_reader='vlm' only; other modes already read the region):
        # forward it so the box stays searchable.
        apply_text_hint_fallback(
            text_doc,
            text=cand.hint_text,
            confidence=cand.hint_text_confidence,
            profile=profile,
            rules=rules,
        )
        resolved.append(_box_with_resolved_text(box, text_doc))

    # `auto_confirmed`: >=1 accepted box AND every accepted box passes the
    # 2-of-2 auto-confirm policy. `verified`: the VLM confirmed a region
    # (at least one accepted box), not merely "a reply was received".
    auto_confirmed = await boxes_auto_confirmed(resolved, candidates, profile)
    return CombinedResolution(
        outcome='resolved',
        boxes=resolved,
        status=verdict_status,
        extra={
            **(class_update or {}),
            **item_verification_fields(
                verified=any(b.state == 'accepted' for b in resolved),
                verifier=vlm_model,
                auto_confirmed=auto_confirmed,
            ),
        },
        trace=trace,
        bbox_wrong=any(b.rejection_reason == REJECT_REASON_VERIFIER for b in boxes),
    )


__all__ = [
    'CLASSIFIER_HIGH_CONF_THRESHOLD',
    'CombinedResolution',
    'resolve_combined_reply',
    'should_classify',
]

"""Region-text OCR reader + item-text reader for the curation detection worker.

See ``scripts/curation/region_worker_main.py`` for the entry point.

Two OCR jobs ride on the region cascade:

* **Region text** -- once a region box is accepted, cut it from the item
  crop (``DetectionProfile.text_crop_margin``), read it with the OCR
  pipeline and keep the dominant text
  (:mod:`src.services.detection.region_text`). The profile's
  ``text_reader`` decides whether that reading, the VLM's, or both land on
  the region.
* **Item text** -- every OCR line on the whole item crop, stored for
  search (:mod:`src.services.curation.item_text`). The same read feeds the
  text-hint step, so the item crop is OCR'd once per pass.

Also the no-VLM path: a deployment without an image LLM accepts detector
regions unverified and fills their text via OCR.

A text-free profile (``text_reader='none'``) stores no region text: the
region-text helpers here strip any text fields from the write instead.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from scripts.curation.worker.cascade import _crop_region_jpeg, _expand_bbox
from scripts.curation.worker.verify import (
    candidate_actor,
    candidate_box,
    chain_entry,
    item_verification_fields,
)
from src.config import get_region_fields
from src.config.region_rejection import REJECT_REASON_SANITY_PREFIX
from src.config.region_source import (
    CANDIDATE_DETECTOR,
    CANDIDATE_DETECTOR_EXISTING,
    CANDIDATE_SEGMENTER,
    CANDIDATE_SEGMENTER_TEXT_HINT,
)
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.item_text import item_text_update
from src.services.curation.region_boxes import has_human_text, new_box_placeholder
from src.services.detection.cascade_detect import is_plausible_region_bbox
from src.services.detection.region_text import (
    TEXT_CHOICE_OCR_ONLY,
    TEXT_CHOICE_VLM_INVALID,
    TEXT_SOURCE_OCR,
    DominantTextConfig,
    DominantTextReading,
    OcrLine,
    ocr_engine_id,
    ocr_needed,
    read_dominant_text,
    resolve_region_text,
)
from src.services.detection.region_text_rules import RegionTextRules, region_text_rules


if TYPE_CHECKING:
    from scripts.curation.worker.state import _ItemTask
    from src.config import DetectionProfile
    from src.services.curation.region_boxes import RegionBox
    from src.services.detection.cascade_detect import PaddleOcrTextRecognizer


logger = get_logger('curation_worker')

# Chain event for a region written without VLM verification because the
# deployment has no VLM configured.
ACCEPTED_UNVERIFIED = 'accepted_unverified'


def candidate_detector(t: _ItemTask, profile: DetectionProfile) -> tuple[str, str]:
    """``(detector, version)`` provenance for the task's candidate box."""
    seg = (profile.segmenter_name, profile.segmenter_version)
    det = (profile.detector_model, profile.detector_version)
    return {
        CANDIDATE_SEGMENTER: seg,
        # OCR-hinted re-pass: the box still comes from the segmenter (on a
        # tighter sub-crop); the text hint lives on the chain.
        CANDIDATE_SEGMENTER_TEXT_HINT: seg,
        CANDIDATE_DETECTOR: det,
        CANDIDATE_DETECTOR_EXISTING: det,
    }.get(t.candidate_source, (t.candidate_source or 'unknown', '1'))


async def read_item_lines(
    ocr: PaddleOcrTextRecognizer, crop_jpeg: bytes, crop_id: str
) -> list[OcrLine] | None:
    """All OCR lines on the item crop; ``None`` when the read failed (the
    caller then writes no item text, so the next pass retries)."""
    try:
        return await ocr.read_lines(crop_jpeg)
    except Exception as exc:
        logger.warning('item_text_ocr_failed', crop_id=crop_id, error=str(exc))
        return None


def item_text_fields(lines: list[OcrLine] | None, *, min_confidence: float) -> dict[str, Any]:
    """Item-text fields for a write; empty when the read failed."""
    if lines is None:
        return {}
    return item_text_update(lines, min_confidence=min_confidence)


async def read_region_text(
    ocr: PaddleOcrTextRecognizer,
    crop_jpeg: bytes,
    region_in_crop: tuple[float, float, float, float],
    profile: DetectionProfile,
    crop_id: str,
) -> DominantTextReading | None:
    """OCR the region (cut from the item crop) and pick its dominant text.

    ``None`` when the OCR call failed -- distinct from a reading with no
    text, which is a real "nothing legible" answer.
    """
    margin = max(0.0, profile.text_crop_margin)
    box = _expand_bbox(region_in_crop, 1.0 + 2.0 * margin) if margin else region_in_crop
    try:
        region_jpeg = _crop_region_jpeg(crop_jpeg, box)
        lines = await ocr.read_region_lines(region_jpeg, min_height=profile.text_crop_min_height)
    except Exception as exc:
        logger.warning('region_text_ocr_failed', crop_id=crop_id, error=str(exc))
        return None
    return read_dominant_text(lines, DominantTextConfig.from_profile(profile))


_TEXT_ATTRS = (
    'text',
    'text_raw',
    'text_source',
    'text_confidence',
    'text_engine_version',
    'text_vlm',
    'text_ocr',
    'text_disagreement',
    'text_choice',
    'text_vlm_invalid',
)


def _drop_region_text(doc: dict[str, Any]) -> None:
    for attr in _TEXT_ATTRS:
        doc.pop(attr, None)


async def apply_region_text(
    doc: dict[str, Any],
    *,
    ocr: PaddleOcrTextRecognizer,
    crop_jpeg: bytes | None,
    region_in_crop: tuple[float, float, float, float],
    profile: DetectionProfile,
    crop_id: str,
    vlm_text: str | None,
    vlm_confidence: str | None,
    vlm_available: bool,
    vlm_model: str | None,
    rules: RegionTextRules | None = None,
) -> None:
    """Replace ``doc``'s region text attributes (keyed by ``RegionBox``
    attribute name) with the profile's text-reader verdict for this region
    (see :func:`resolve_region_text`).

    ``rules`` (default: the profile's, with the resolved prompt pack's
    examples) decide which readings are text at all; a VLM reading they
    reject counts as no reading, so the OCR reader runs in
    ``vlm_then_ocr`` mode too. A text-free profile reads nothing and
    leaves ``doc`` with no region text fields.

    ``vlm_model`` is the resolved model of the VLM that produced
    ``vlm_text`` (``None`` when no VLM ran): it is the reading's engine
    version.
    """
    if not profile.reads_text:
        _drop_region_text(doc)
        return
    rules = rules or region_text_rules(profile)
    vlm_usable = vlm_text if vlm_text and rules.invalid_reason(vlm_text) is None else None
    reading = None
    if crop_jpeg is not None and ocr_needed(
        profile.text_reader, vlm_text=vlm_usable, vlm_available=vlm_available
    ):
        reading = await read_region_text(ocr, crop_jpeg, region_in_crop, profile, crop_id)
    fields = resolve_region_text(
        profile.text_reader,
        vlm_text=vlm_text,
        vlm_confidence=vlm_confidence,
        vlm_engine=vlm_model or '',
        ocr=reading,
        ocr_engine=ocr_engine_id(profile),
        normalizer=DominantTextConfig.from_profile(profile).normalizer,
        rules=rules,
    )
    _drop_region_text(doc)
    doc.update(fields)


_TEXT_BOX_ATTRS = (
    'text',
    'text_raw',
    'text_source',
    'text_engine_version',
    'text_confidence',
    'text_vlm',
    'text_ocr',
    'text_choice',
    'text_vlm_invalid',
    'text_disagreement',
)


def _box_with_resolved_text(box: RegionBox, doc: dict[str, Any]) -> RegionBox:
    """Replace one box's text-ish attributes with :func:`apply_region_text`'s
    verdict, whatever it decided (including nothing at all).

    ``doc`` is keyed by ``RegionBox`` attribute name (``text`` etc). Every
    text attribute is reset to ``None`` first, then overwritten with
    whatever ``doc`` supplies -- ``apply_region_text`` is authoritative
    here (its own ``_drop_region_text`` + fresh-set behavior), so a
    text-free profile's empty ``doc`` clears any text
    :func:`~scripts.curation.worker.verify.verdicts_to_boxes` had
    provisionally set on the box from the raw VLM verdict, rather than
    leaving it in place.
    """
    if has_human_text(box):
        return box
    updates: dict[str, Any] = dict.fromkeys(_TEXT_BOX_ATTRS)
    updates.update({attr: doc[attr] for attr in _TEXT_BOX_ATTRS if attr in doc})
    return dataclasses.replace(box, **updates)


def resolve_rejected_box_text(
    box: RegionBox,
    *,
    profile: DetectionProfile,
    rules: RegionTextRules | None = None,
) -> RegionBox:
    """W8 M6 fix: a non-accepted box must never carry a raw, unvalidated
    VLM text reply the way an accepted box's ``apply_region_text`` output
    is validated.

    ``verdicts_to_boxes`` provisionally sets a rejected/no-verdict box's
    ``text`` straight from the VLM's own reply (or the candidate's OCR
    hint) with no profile/rules gate at all -- the accepted-box path's
    ``_box_with_resolved_text`` fix only ever runs on accepted boxes. A
    text-free profile (``reads_text=False``) drops it entirely, same as
    an accepted box would; a text-reading profile keeps it only if it
    passes the same :func:`~src.services.detection.region_text_rules
    .region_text_rules` an accepted box's VLM reading is held to.
    """
    if has_human_text(box):
        return box
    if not profile.reads_text:
        return _box_with_resolved_text(box, {})
    if box.text is None:
        return box
    rules = rules or region_text_rules(profile)
    if rules.invalid_reason(box.text) is not None:
        return dataclasses.replace(box, text=None)
    return box


async def accept_without_vlm(
    t: _ItemTask,
    *,
    ocr: PaddleOcrTextRecognizer,
    profile: DetectionProfile,
    rules: RegionTextRules | None = None,
) -> None:
    """No VLM configured: write every candidate region unverified.

    W8: writes the box-list shape. With no VLM to adjudicate, each candidate
    (a re-verified stored ``proposed`` box keeps its id, so none stays
    ``proposed`` forever) is accepted if it passes the same geometry gate the
    verified path uses, else rejected with the sanity reason. The item lands
    ``detected`` when any box is accepted, ``detection_failed`` when none
    is. Accepted text comes from OCR (none on a text-free profile); a human's
    typed text is kept. The verifier/verified fields stay unset: the region
    is still reviewable in the human region queue.
    """
    if not t.candidates or t.crop_jpeg is None:
        msg = f'accept_without_vlm needs at least one candidate and crop bytes (crop {t.crop_id})'
        raise ValueError(msg)
    F = get_region_fields()
    boxes: list[RegionBox] = []
    for i, cand in enumerate(t.candidates):
        actor = candidate_actor(cand)
        for entry in chain_entry(actor, 'hit'):
            if entry not in t.detection_trace:
                t.detection_trace.append(entry)
        gate_ok, gate_reason = is_plausible_region_bbox(cand.bbox_in_crop, t.item_bbox_norm)
        if not gate_ok:
            t.detection_trace.extend(chain_entry(actor, f'sanity_reject:{gate_reason}'))
            boxes.append(
                candidate_box(
                    cand,
                    fallback_id=new_box_placeholder(i),
                    state='rejected',
                    rejection_reason=f'{REJECT_REASON_SANITY_PREFIX}{gate_reason}',
                )
            )
            continue
        box = candidate_box(cand, fallback_id=new_box_placeholder(i), state='accepted')
        if not has_human_text(box):
            text_doc: dict[str, Any] = {}
            await apply_region_text(
                text_doc,
                ocr=ocr,
                crop_jpeg=t.crop_jpeg,
                region_in_crop=cand.bbox_in_crop,
                profile=profile,
                crop_id=t.crop_id,
                vlm_text=None,
                vlm_confidence=None,
                vlm_available=False,
                vlm_model=None,
                rules=rules,
            )
            box = _box_with_resolved_text(box, text_doc)
        boxes.append(box)
    if any(b.state == 'accepted' for b in boxes):
        t.detection_trace.extend(chain_entry(candidate_actor(t.candidates[0]), ACCEPTED_UNVERIFIED))
        status = RegionStatus.DETECTED
        t.pending_empty_status = RegionStatus.DETECTED
    else:
        status = RegionStatus.DETECTION_FAILED
        t.pending_status = RegionStatus.DETECTION_FAILED
    # The box list and its real status are finished at write time against
    # the live doc (``bulk_writer._merge``); ``status`` here is the
    # provisional value ``region_embed_stage`` / ``_publish_region_events``
    # read from ``update_doc`` before that merge (see ``runner._box_list_doc``).
    t.pending_boxes = boxes
    t.update_doc = {
        F.status: status,
        F.detector_chain: list(t.detection_trace),
        **item_verification_fields(verified=False, verifier=None),
    }


def apply_text_hint_fallback(
    doc: dict[str, Any],
    *,
    text: str | None,
    confidence: float | None,
    profile: DetectionProfile,
    rules: RegionTextRules,
) -> None:
    """Forward the item-crop OCR text that seeded a text-hint box when the
    region itself got no text -- if that text passes ``rules``. ``doc`` is
    keyed by ``RegionBox`` attribute name, like :func:`apply_region_text`'s."""
    if not profile.reads_text:
        _drop_region_text(doc)
        return
    if doc.get('text') or not text or rules.invalid_reason(text) is not None:
        return
    doc['text'] = text
    doc['text_raw'] = text
    doc['text_source'] = TEXT_SOURCE_OCR
    doc['text_engine_version'] = ocr_engine_id(profile)
    doc['text_confidence'] = confidence
    doc['text_choice'] = (
        TEXT_CHOICE_VLM_INVALID if doc.get('text_vlm_invalid') else TEXT_CHOICE_OCR_ONLY
    )


__all__ = [
    'ACCEPTED_UNVERIFIED',
    'accept_without_vlm',
    'apply_region_text',
    'apply_text_hint_fallback',
    'candidate_detector',
    'item_text_fields',
    'read_item_lines',
    'read_region_text',
    'resolve_rejected_box_text',
]

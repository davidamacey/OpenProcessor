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

from typing import TYPE_CHECKING, Any

from scripts.curation.worker.cascade import _crop_region_jpeg, _expand_bbox
from scripts.curation.worker.verify import _region_reject_doc, _region_write_doc
from src.config import get_region_fields
from src.config.region_rejection import REJECT_REASON_SANITY_PREFIX
from src.config.region_source import (
    CANDIDATE_DETECTOR,
    CANDIDATE_DETECTOR_EXISTING,
    CANDIDATE_SEGMENTER,
    CANDIDATE_SEGMENTER_TEXT_HINT,
)
from src.core.logging import get_logger
from src.services.curation.item_text import item_text_update
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
from src.services.labeling.vlm_client import DEFAULT_MODEL as VLM_MODEL_ID


if TYPE_CHECKING:
    from scripts.curation.worker.state import _ItemTask
    from src.config import DetectionProfile
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
    F = get_region_fields()
    for attr in _TEXT_ATTRS:
        doc.pop(getattr(F, attr), None)


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
    rules: RegionTextRules | None = None,
) -> None:
    """Replace ``doc``'s region text fields with the profile's text-reader
    verdict for this region (see :func:`resolve_region_text`).

    ``rules`` (default: the profile's, with the resolved prompt pack's
    examples) decide which readings are text at all; a VLM reading they
    reject counts as no reading, so the OCR reader runs in
    ``vlm_then_ocr`` mode too. A text-free profile reads nothing and
    leaves ``doc`` with no region text fields.
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
        vlm_engine=VLM_MODEL_ID,
        ocr=reading,
        ocr_engine=ocr_engine_id(profile),
        normalizer=DominantTextConfig.from_profile(profile).normalizer,
        rules=rules,
    )
    _drop_region_text(doc)
    F = get_region_fields()
    for attr, value in fields.items():
        doc[getattr(F, attr)] = value


async def accept_without_vlm(
    t: _ItemTask,
    *,
    ocr: PaddleOcrTextRecognizer,
    profile: DetectionProfile,
    rules: RegionTextRules | None = None,
) -> None:
    """No VLM configured: write the task's candidate region unverified.

    The box passes the same geometry gate the verified path uses; the
    region lands ``detected`` with ``verified=False`` / ``validated=False``
    (so human review still sees it) and its text comes from OCR (none on a
    text-free profile).
    """
    if t.candidate_in_crop is None or t.candidate_in_source is None or t.crop_jpeg is None:
        msg = f'accept_without_vlm needs a candidate box and crop bytes (crop {t.crop_id})'
        raise ValueError(msg)
    actor, version = candidate_detector(t, profile)
    if f'{actor}:hit' not in t.detection_trace:
        t.detection_trace.append(f'{actor}:hit')
    gate_ok, gate_reason = is_plausible_region_bbox(t.candidate_in_crop, t.item_bbox_norm)
    if not gate_ok:
        t.detection_trace.append(f'{actor}:sanity_reject:{gate_reason}')
        t.update_doc = _region_reject_doc(
            detector=actor,
            detector_version=version,
            reason=f'{REJECT_REASON_SANITY_PREFIX}{gate_reason}',
            chain=t.detection_trace,
        )
        return
    t.detection_trace.append(f'{actor}:{ACCEPTED_UNVERIFIED}')
    doc = _region_write_doc(
        region_in_source=t.candidate_in_source,
        score=t.candidate_score,
        detector=actor,
        detector_version=version,
        chain=t.detection_trace,
        region_verified=False,
        verifier=None,
        verifier_version=None,
    )
    doc[get_region_fields().source] = t.candidate_source
    await apply_region_text(
        doc,
        ocr=ocr,
        crop_jpeg=t.crop_jpeg,
        region_in_crop=t.candidate_in_crop,
        profile=profile,
        crop_id=t.crop_id,
        vlm_text=None,
        vlm_confidence=None,
        vlm_available=False,
        rules=rules,
    )
    t.update_doc = doc


def apply_text_hint_fallback(
    doc: dict[str, Any],
    *,
    text: str | None,
    confidence: float | None,
    profile: DetectionProfile,
    rules: RegionTextRules,
) -> None:
    """Forward the item-crop OCR text that seeded a text-hint box when the
    region itself got no text -- if that text passes ``rules``."""
    if not profile.reads_text:
        _drop_region_text(doc)
        return
    F = get_region_fields()
    if doc.get(F.text) or not text or rules.invalid_reason(text) is not None:
        return
    doc[F.text] = text
    doc[F.text_raw] = text
    doc[F.text_source] = TEXT_SOURCE_OCR
    doc[F.text_engine_version] = ocr_engine_id(profile)
    doc[F.text_confidence] = confidence
    doc[F.text_choice] = (
        TEXT_CHOICE_VLM_INVALID if doc.get(F.text_vlm_invalid) else TEXT_CHOICE_OCR_ONLY
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
]

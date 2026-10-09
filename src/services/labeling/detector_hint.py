"""Optional per-item detector-name hint for the VLM prompt (#193).

On the public COCO oracle the stored detector name was right about 96.6% of the time
against 72.7% for the VLM, so a pack can opt in to telling the VLM what the detector
saw. It is a hint only: the reply parser still accepts any answer, and the stored
detector fields are read, never written. Class identity is the name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.labeling.vlm_models import ItemCrop


if TYPE_CHECKING:
    from src.services.labeling.vlm_prompts import PromptPack


#: Item fields a caller must read to build a hint.
DETECTOR_HINT_FIELDS = ('detector_class_name', 'detector_confidence')

#: ``detector_hint_min_confidence_pct`` bounds; 0 is off.
MAX_DETECTOR_HINT_PCT = 100


def detector_hint_for(src: dict[str, Any], min_confidence_pct: int) -> tuple[str, float | None]:
    """The ``(detector class name, confidence)`` to hint, or ``('', None)``.

    Empty when the setting is off (0), the item has no stored detector name or
    confidence, or the confidence is below ``min_confidence_pct`` percent.
    """
    if min_confidence_pct <= 0:
        return '', None
    name = src.get('detector_class_name')
    conf = src.get('detector_confidence')
    if not isinstance(name, str) or not name:
        return '', None
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return '', None
    if conf * 100 < min_confidence_pct:
        return '', None
    return name, float(conf)


def hinted_crop(crop_id: str, jpeg: bytes, src: dict[str, Any], pack: PromptPack) -> ItemCrop:
    """An :class:`ItemCrop` carrying the detector hint ``pack`` asks for, if any."""
    name, conf = detector_hint_for(src, pack.detector_hint_min_confidence_pct)
    return ItemCrop(img_id=crop_id, jpeg_bytes=jpeg, detector_class=name, detector_confidence=conf)

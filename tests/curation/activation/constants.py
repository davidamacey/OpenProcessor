"""Shared routes and request-body builders for the activation tests."""

from __future__ import annotations

from typing import Any

from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


PREFIX = '/curation/projects/default/prompt_packs'
RP = '/curation/projects/default/region_profiles'
SETTINGS = '/curation/projects/default/settings'


def pack_body(class_system: str | None = None) -> dict[str, Any]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    if class_system is not None:
        body['class_system'] = class_system
    return body


def stripped_pack_body() -> dict[str, Any]:
    """A pack whose combined prompts lack the multi-box keys, so activating it under a
    multi-region profile fails the never-bypassable gate."""
    body = pack_body('STRIPPED')
    body['combined_system'] = (
        'Return JSON keys class_id class_confidence region_visible region_bbox_correct '
        'region_confidence'
    )
    body['combined_batch_system'] = (
        'Return JSON results with img class_id class_confidence region_visible '
        'region_bbox_correct region_confidence'
    )
    return body


def live_pair():
    from src.services.detection.profile_registry import get_active_region_profile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    return active_prompt_pack(), get_active_region_profile()

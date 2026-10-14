"""Item fields that record a VLM class attempt, and their index mapping."""

from __future__ import annotations

from typing import Any


VLM_CLASS_ATTEMPTED_AT_FIELD = 'vlm_class_attempted_at'
VLM_CLASS_EMPTY_REASON_FIELD = 'vlm_class_empty_reason'

VLM_CLASS_ATTEMPT_MAPPING: dict[str, Any] = {
    VLM_CLASS_ATTEMPTED_AT_FIELD: {'type': 'date'},
    VLM_CLASS_EMPTY_REASON_FIELD: {'type': 'keyword'},
}

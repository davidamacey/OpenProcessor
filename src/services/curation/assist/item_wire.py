"""The item's ``assist`` object on the wire."""

from __future__ import annotations

from typing import Any


def _attributes(raw: Any) -> list[dict[str, Any]]:
    return [
        {
            'key': a.get('key'),
            'value': a.get('value'),
            'confidence': a.get('confidence'),
        }
        for a in raw or []
        if isinstance(a, dict)
    ]


def assist_to_wire(raw: Any, box_ids: set[str]) -> dict[str, Any] | None:
    """The stored ``assist`` object as the wire carries it, or ``None``. Box
    suggestions for a box id the item no longer has are dropped."""
    if not isinstance(raw, dict) or not raw:
        return None
    return {
        'run_id': raw.get('run_id'),
        'run_created_at': raw.get('run_created_at'),
        'vlm_endpoint': raw.get('vlm_endpoint'),
        'vlm_model': raw.get('vlm_model'),
        'prompt_pack': raw.get('prompt_pack'),
        'prompt_pack_revision': raw.get('prompt_pack_revision'),
        'class_name': raw.get('class_name'),
        'class_confidence': raw.get('class_confidence'),
        'attributes': _attributes(raw.get('attributes')),
        'boxes': [
            {'box_id': b.get('box_id'), 'attributes': _attributes(b.get('attributes'))}
            for b in raw.get('boxes') or []
            if isinstance(b, dict) and b.get('box_id') in box_ids
        ],
    }


__all__ = ['assist_to_wire']

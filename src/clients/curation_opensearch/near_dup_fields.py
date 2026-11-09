"""The near-duplicate fields both curation indexes carry (images and items): the search
projection of the current run's groups (research plan 5.3). The projection sync writes them;
`hide_near_duplicates` reads `near_dup_is_representative`.
"""

from __future__ import annotations

from typing import Any


NEAR_DUP_MAPPING: dict[str, Any] = {
    'near_dup_group_id': {'type': 'keyword'},
    'near_dup_leak_group_id': {'type': 'keyword'},
    'near_dup_is_representative': {'type': 'boolean'},
    'near_dup_group_size': {'type': 'integer'},
    'near_dup_run_id': {'type': 'keyword'},
}


def null_safe_floor(field: str, floor: float) -> dict[str, Any]:
    """``field >= floor``, but an item without the field is kept."""
    return {
        'bool': {
            'should': [
                {'range': {field: {'gte': floor}}},
                {'bool': {'must_not': {'exists': {'field': field}}}},
            ],
            'minimum_should_match': 1,
        }
    }


def near_dup_visible_clause() -> dict[str, Any]:
    """Hides only items a scoring pass marked as a non-representative duplicate;
    never-scored items stay."""
    return {
        'bool': {
            'should': [
                {'term': {'near_dup_is_representative': True}},
                {'bool': {'must_not': {'exists': {'field': 'near_dup_is_representative'}}}},
            ],
            'minimum_should_match': 1,
        }
    }

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

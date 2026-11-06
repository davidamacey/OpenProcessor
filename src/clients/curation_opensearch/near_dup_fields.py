"""The near-duplicate fields of the ``items`` body: the group an item belongs to and how the
group was scored. Lane ND-A owns this module.
"""

from __future__ import annotations

from typing import Any


NEAR_DUP_ITEM_MAPPING: dict[str, Any] = {
    'dup_group_id': {'type': 'keyword'},
    'dup_group_size': {'type': 'integer'},
    'dup_is_representative': {'type': 'boolean'},
    'dup_threshold': {'type': 'float'},
    'dup_method': {'type': 'keyword'},
    'dup_scored_at': {'type': 'date'},
}

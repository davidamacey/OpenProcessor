"""What an item's stored box means: who set it and what a person may change.

Pure rules shared by the lock predicate (:mod:`src.clients.occ_locks`), the drawn-item
create and the geometry edit. ``bbox_norm`` is the box in image coordinates
(``x1, y1, x2, y2``, each in ``[0, 1]``); only the item geometry writers change it for a
person, and an automated writer never changes a box a person set.
"""

from __future__ import annotations

from typing import Any


def stored_box(value: Any) -> tuple[float, float, float, float] | None:
    """A stored 4-number box as a tuple, or ``None`` when the field is absent or malformed."""
    if not value or len(value) != 4:
        return None
    return (value[0], value[1], value[2], value[3])


GEOMETRY_SOURCE_HUMAN = 'human'

GEOMETRY_OWNED_FIELDS = ('bbox_norm', 'crop_area_norm')
"""The item fields that describe the box; an automated writer never overwrites them on an item
a person drew, moved or deleted."""

NOT_AN_OBJECT = 'not_an_object'
"""The exclude reason of a person's soft delete: the box was never an object. The item stays a
document so its box keeps blocking a re-detect from creating it again."""

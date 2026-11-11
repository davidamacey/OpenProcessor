"""What an item's stored box means: who set it and what a person may change.

Pure rules shared by the lock predicate (:mod:`src.clients.occ_locks`), the drawn-item
create and the geometry edit. ``bbox_norm`` is the box in image coordinates
(``x1, y1, x2, y2``, each in ``[0, 1]``); only the item geometry writers change it for a
person, and an automated writer never changes a box a person set.
"""

from __future__ import annotations

import math
from hashlib import sha256
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence


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


MIN_BOX_SIDE_NORM = 0.004
"""The shortest side a drawn or edited box may have, as a fraction of the image side."""


class ItemBoxError(ValueError):
    """A drawn or edited box is refused. ``code`` is the API error code; ``fields`` are the
    extra values the error body carries (``min_side`` for ``box_too_small``)."""

    def __init__(self, code: str, message: str, **fields: float) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fields = fields


def validate_item_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    """``box`` as ``(x1, y1, x2, y2)`` in image coordinates, or :class:`ItemBoxError`."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
        raise ItemBoxError('box_out_of_image', 'the box must lie inside the image')
    if x2 <= x1 or y2 <= y1:
        raise ItemBoxError('degenerate_box', 'the box needs x1 < x2 and y1 < y2')
    if min(x2 - x1, y2 - y1) < MIN_BOX_SIDE_NORM:
        raise ItemBoxError(
            'box_too_small',
            f'a side of the box is shorter than {MIN_BOX_SIDE_NORM} of the image',
            min_side=MIN_BOX_SIDE_NORM,
        )
    return (x1, y1, x2, y2)


def box_area_norm(box: Sequence[float]) -> float:
    """The stored ``crop_area_norm`` of a box."""
    return round((box[2] - box[0]) * (box[3] - box[1]), 6)


def drawn_crop_id(image_id: str, request_id: uuid.UUID) -> str:
    """The id of the item drawn by request ``request_id`` on ``image_id``. A retry of the same
    request lands on the same document, which is what makes the create idempotent."""
    return sha256(f'{image_id}|drawn|{request_id}'.encode()).hexdigest()[:32]

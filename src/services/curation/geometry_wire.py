"""What the wire says about an item's box: who set it, whether an automated writer may touch it,
and the text to show for that. The lock is the predicate the writers use
(:func:`~src.clients.occ_locks.is_locked_geometry`), so the two cannot disagree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from src.clients.occ_locks import (
    has_human_geometry,
    has_import_geometry,
    is_human_deleted,
    is_locked_geometry,
)
from src.domain.region_boxes import is_human_owned


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.domain.region_boxes import RegionBox

GeometrySource = Literal['detector', 'human', 'import']

LABEL_HUMAN = 'Set by a person'
LABEL_DELETED = 'Deleted by a person'
LABEL_IMPORT = 'Set by an import'
LABEL_HOLDOUT = 'In the test holdout'


def geometry_source(src: Mapping[str, Any]) -> GeometrySource:
    if has_human_geometry(src):
        return 'human'
    return 'import' if has_import_geometry(src) else 'detector'


def geometry_lock_label(src: Mapping[str, Any]) -> str | None:
    """The served reason the box is locked; ``None`` when it is not."""
    if is_human_deleted(src):
        return LABEL_DELETED
    if has_human_geometry(src):
        return LABEL_HUMAN
    if has_import_geometry(src):
        return LABEL_IMPORT
    return LABEL_HOLDOUT if src.get('test_holdout') else None


def geometry_wire(src: Mapping[str, Any]) -> dict[str, Any]:
    """The geometry fields of a wire item."""
    return {
        'geometry_revision': int(src.get('geometry_revision') or 0),
        'geometry_source': geometry_source(src),
        'geometry_locked': is_locked_geometry(src),
        'lock_label': geometry_lock_label(src),
    }


def region_box_lock_label(box: RegionBox, *, locked: bool) -> str | None:
    """The served reason a region box is locked (``locked`` is ``is_locked_box``)."""
    if not locked:
        return None
    return LABEL_HUMAN if is_human_owned(box) else LABEL_IMPORT

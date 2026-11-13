"""What the wire says about an item's box: who set it, whether the editor may move it, and the
text to show for that. ``geometry_locked`` is the editor's lock, not the automated writers' rule
(:func:`~src.clients.occ_locks.is_locked_geometry`): a person may always move, resize or delete
a box a person drew or edited, so human geometry locks automated writers but not the editor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from src.clients.occ_locks import has_human_geometry, has_import_geometry, is_human_deleted
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


def is_editor_locked(src: Mapping[str, Any]) -> bool:
    """True when the editor must not move the box: an import wrote it, it is in the test
    holdout, or a person deleted it. A person's own geometry stays editable."""
    return has_import_geometry(src) or bool(src.get('test_holdout')) or is_human_deleted(src)


def geometry_wire(src: Mapping[str, Any]) -> dict[str, Any]:
    """The geometry fields of a wire item."""
    return {
        'geometry_revision': int(src.get('geometry_revision') or 0),
        'geometry_source': geometry_source(src),
        'geometry_locked': is_editor_locked(src),
        'lock_label': geometry_lock_label(src),
    }


def region_box_lock_label(box: RegionBox, *, locked: bool) -> str | None:
    """The served reason a region box is locked (``locked`` is ``is_locked_box``)."""
    if not locked:
        return None
    return LABEL_HUMAN if is_human_owned(box) else LABEL_IMPORT

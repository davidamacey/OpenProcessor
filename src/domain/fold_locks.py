"""The wire's lock flags, evaluated on folded state (15a section 4.4).

``label_locked`` and a box's ``locked`` keep today's predicates (``src/clients/occ_locks.py``) so
the screens read the same answers; they are computed on the folded row, not stored. Writers do not
consult them any more: a machine write can no longer change what a claim decides, so the lock rule
is the shape of the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.region_rejection import REJECT_REASON_HUMAN
from src.config.region_source import CANDIDATE_IMPORT
from src.domain.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE


if TYPE_CHECKING:
    from collections.abc import Iterable


def is_human_marker(value: object) -> bool:
    """A source string that names a person (``human``, ``human_move``, ...)."""
    return isinstance(value, str) and 'human' in value


def box_is_locked(
    *, source: str | None, rejection_reason: str | None, text_source: str | None, state: str
) -> bool:
    """A person created, rejected or typed the box, or an import wrote it and it is not a mere
    suggestion (``is_locked_box``). A person's accept of a machine box leaves no trace."""
    human = source == 'human' or rejection_reason == REJECT_REASON_HUMAN or text_source == 'human'
    return human or (source == CANDIDATE_IMPORT and state != 'proposed')


def is_locked_class(*, class_source: str | None, class_validated: bool, holdout: bool) -> bool:
    return (
        is_human_marker(class_source)
        or (class_source == LABEL_IMPORT_CLASS_SOURCE and class_validated)
        or holdout
    )


def has_import_geometry(
    *, import_ids: Iterable[str], class_source: str | None, class_validated: bool
) -> bool:
    return bool(list(import_ids)) and class_source == LABEL_IMPORT_CLASS_SOURCE and class_validated


def is_locked_geometry(*, human_geometry: bool, import_geometry: bool, holdout: bool) -> bool:
    return human_geometry or import_geometry or holdout


def is_editor_locked(*, import_geometry: bool, holdout: bool, deleted: bool) -> bool:
    """The editor may not move the box: an import wrote it, it is held out, or it was deleted."""
    return import_geometry or holdout or deleted


def is_locked_item(
    *,
    class_locked: bool,
    geometry_locked: bool,
    any_box_locked: bool,
    region_validated: bool,
    region_verifier: str | None,
) -> bool:
    return (
        class_locked
        or geometry_locked
        or any_box_locked
        or (region_validated and region_verifier in ('human', 'import'))
    )

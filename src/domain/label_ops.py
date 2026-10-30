"""The pure rules of a label operation: its id, what it wrote, and the per-item
decision an undo makes.

An operation is one bulk human class write (batch label or move). Undoing it
restores each item whose next undo step is that operation and leaves every other
item alone. Imports only ``src.domain`` and ``src.utils``.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from src.domain.history import find_undo_snapshot, op_write_undone
from src.domain.ingest_class_sources import HUMAN_CLASS_SOURCE
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

LabelOpKind = Literal['label', 'move']

LABEL_OP_ID_RE = re.compile(r'lop_\d{8}T\d{12}_[0-9a-f]{6}')
"""Matched with ``fullmatch`` before any path is built from an id."""

HUMAN_MOVE_CLASS_SOURCE = 'human_move'


def new_label_op_id(now: datetime) -> str:
    """``lop_<UTC YYYYMMDDTHHMMSSffffff>_<6 hex>``; sorts in creation order."""
    return f'lop_{now.strftime("%Y%m%dT%H%M%S%f")}_{secrets.token_hex(3)}'


class UndoOutcome(str, Enum):  # noqa: UP042 -- str-valued members are serialised as the plain count keys
    RESTORE = 'restore'
    ALREADY_UNDONE = 'already_undone'
    KEPT_CHANGED = 'kept_changed'
    KEPT_HOLDOUT = 'kept_holdout'
    KEPT_STALE_CLASS = 'kept_stale_class'


@dataclass(frozen=True)
class OpFingerprint:
    """What the operation wrote on every item. ``None`` means "not compared"."""

    op_id: str
    writer: str
    at: str
    class_name: str | None
    class_source: str | None
    class_validated: bool
    cluster_id: int | None


def label_fingerprint(*, op_id: str, writer: str, at: str, class_name: str) -> OpFingerprint:
    return OpFingerprint(op_id, writer, at, class_name, HUMAN_CLASS_SOURCE, True, None)


def move_fingerprint(
    *, op_id: str, writer: str, at: str, class_name: str | None, cluster_id: int | None
) -> OpFingerprint:
    """A move into a class cluster (``class_name`` set) or a candidate cluster
    (``class_name`` ``None``: only the placement and the unvalidated state are compared)."""
    if class_name is not None:
        return OpFingerprint(op_id, writer, at, class_name, HUMAN_MOVE_CLASS_SOURCE, True, None)
    return OpFingerprint(op_id, writer, at, None, None, False, cluster_id)


def _excluded_after(source: dict[str, Any], fp: OpFingerprint) -> bool:
    if not source.get('class_excluded'):
        return False
    stamp = source.get('excluded_at')
    return not (stamp and stamp < fp.at)


def _matches_fingerprint(source: dict[str, Any], fp: OpFingerprint) -> bool:
    if bool(source.get('class_validated')) != fp.class_validated:
        return False
    if fp.class_source is not None and source.get('class_source') != fp.class_source:
        return False
    if fp.cluster_id is not None and source.get('cluster_id') != fp.cluster_id:
        return False
    if fp.class_name is not None:
        return normalize_class_name(source.get('class_name') or '') == normalize_class_name(
            fp.class_name
        )
    return True


def undo_outcome(
    source: dict[str, Any] | None,
    fp: OpFingerprint,
    *,
    registry_name: Callable[[int], str | None],
) -> UndoOutcome:
    """What an undo of the operation does to one item. First matching rule wins."""
    if source is None:
        return UndoOutcome.KEPT_CHANGED
    history = source.get('class_id_history')
    if source.get('label_op_id') != fp.op_id:
        if op_write_undone(history, fp.op_id):
            return UndoOutcome.ALREADY_UNDONE
        return UndoOutcome.KEPT_CHANGED
    if source.get('test_holdout'):
        return UndoOutcome.KEPT_HOLDOUT
    snap = find_undo_snapshot(history)
    if snap is None or snap.get('op_id') != fp.op_id or snap.get('writer') != fp.writer:
        return UndoOutcome.KEPT_CHANGED
    if _excluded_after(source, fp) or not _matches_fingerprint(source, fp):
        return UndoOutcome.KEPT_CHANGED
    restore_id = snap.get('class_id')
    if restore_id is not None and normalize_class_name(
        registry_name(int(restore_id)) or ''
    ) != normalize_class_name(snap.get('class_name') or ''):
        return UndoOutcome.KEPT_STALE_CLASS
    return UndoOutcome.RESTORE


def current_query(fp: OpFingerprint) -> dict[str, Any]:
    """The query-expressible part of :func:`undo_outcome`'s ``RESTORE``: items that
    still carry the operation's label. The stack and registry rules cannot be
    expressed; a tagged item always has the operation on top of its stack, so this
    overcounts only items whose restore target class was renamed or deleted."""
    filters: list[dict[str, Any]] = [
        {'term': {'label_op_id': fp.op_id}},
        {'term': {'class_validated': fp.class_validated}},
    ]
    if fp.class_source is not None:
        filters.append({'term': {'class_source': fp.class_source}})
    if fp.cluster_id is not None:
        filters.append({'term': {'cluster_id': fp.cluster_id}})
    if fp.class_name is not None:
        filters.append({'term': {'class_name': fp.class_name}})
    return {
        'bool': {
            'filter': filters,
            'must_not': [
                {'term': {'test_holdout': True}},
                {
                    'bool': {
                        'must': [{'term': {'class_excluded': True}}],
                        'must_not': [{'range': {'excluded_at': {'lt': fp.at}}}],
                    }
                },
            ],
        }
    }


__all__ = [
    'HUMAN_MOVE_CLASS_SOURCE',
    'LABEL_OP_ID_RE',
    'LabelOpKind',
    'OpFingerprint',
    'UndoOutcome',
    'current_query',
    'label_fingerprint',
    'move_fingerprint',
    'new_label_op_id',
    'undo_outcome',
]

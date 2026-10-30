"""Wire models for label operations (bulk label and move, recorded so they can be undone by id)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class LabelOpUndoPass(BaseModel):
    """One completed undo pass over an operation."""

    at: str
    restored: int
    already_undone: int
    kept_changed: int
    kept_holdout: int
    kept_stale_class: int
    conflicts: int


class LabelOp(BaseModel):
    """One recorded bulk label or move. Class ids are never served: identity is the name."""

    op_id: str
    kind: Literal['label', 'move']
    at: str
    finished_at: str | None
    class_name: str | None
    target_cluster_id: int | None
    target_kind: Literal['class', 'candidate'] | None
    label_source: str | None
    target: Literal['crop_ids', 'selection']
    selected: int
    count: int | None
    conflicts: int | None
    current: int
    status: Literal['writing', 'interrupted', 'written', 'undoing', 'undo_interrupted', 'undone']
    undone: bool
    undoable: bool
    last_undo: LabelOpUndoPass | None


class LabelOpList(BaseModel):
    items: list[LabelOp]
    limit: int
    retention: int


class LabelOpUndoRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dry_run: bool = True  # writing needs an explicit false, like the loop rollback


class LabelOpUndoResponse(BaseModel):
    """What an undo pass did (or, for a dry run, would do) per item.

    ``matched`` is the operation's members and equals ``restored + already_undone +
    kept_changed + kept_holdout + kept_stale_class + conflicts``.
    """

    op_id: str
    dry_run: bool
    status: str
    members_known: bool
    matched: int
    restored: int
    already_undone: int
    kept_changed: int
    kept_holdout: int
    kept_stale_class: int
    conflicts: int
    conflict_ids: list[str]


class CropUndoBatchResponse(BaseModel):
    """``POST /crops/label/undo_batch``."""

    items: list[dict[str, Any]]
    undone: int
    nothing_to_undo: list[str]
    conflicts: list[str]
    not_found: list[str]


__all__ = [
    'CropUndoBatchResponse',
    'LabelOp',
    'LabelOpList',
    'LabelOpUndoPass',
    'LabelOpUndoRequest',
    'LabelOpUndoResponse',
]

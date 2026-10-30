"""Wire models for label operations (bulk label and move, recorded so they can be undone by id)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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


__all__ = ['LabelOp', 'LabelOpList', 'LabelOpUndoPass']

"""Wire models for label operations (bulk label and move, recorded so they can be undone by id)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.domain.label_ops import LabelOpKind


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
    kind: LabelOpKind
    at: str
    finished_at: str | None
    class_name: str | None
    target_cluster_uid: str | None
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
    retention: int = Field(description='Bulk operations kept per project (the newest).')
    single_retention: int = Field(
        description='Single label writes kept per project (the newest), counted apart from the bulk ones.'
    )
    next_after: str | None = Field(default=None, description='Pass as `after` for the next page.')


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


class AdoptDetectorRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dry_run: bool = True  # writing needs an explicit false


class AdoptDetectorResponse(BaseModel):
    """``POST /label_ops/adopt_detector``: what adoption would do (a dry run) or did.

    The counts are by detector name. A real run adds the registry classes the detector's labels
    need, then queues the recorded operation (``op_id`` and ``job_id``, both null when no item was
    left to label)."""

    dry_run: bool
    items: int = Field(description='Items that have a detector class but no class of their own.')
    classes_to_create: list[str] = Field(
        description='Detector names the registry lacks (a dry run) or that were added (a real run).'
    )
    blocked: list[str] = Field(
        description='Detector names of a deprecated class or the region class: skipped.'
    )
    blocked_items: int = Field(description='Items left without a class because of `blocked`.')
    remaining: int = Field(
        description='Items beyond this operation (one writes at most 20,000): adopt again.'
    )
    op_id: str | None = Field(default=None, description='The recorded operation (undo it by id).')
    job_id: str | None = Field(default=None, description='The queued job; poll GET /jobs/{job_id}.')


class AcceptedClassCount(BaseModel):
    class_name: str
    count: int


class AcceptHeldBack(BaseModel):
    """Items the bar matches that the operation will not write, each counted once under the first
    reason that applies, in the order of the fields."""

    excluded: int = Field(description='Excluded, or dismissed from review.')
    holdout: int = Field(description='In the frozen test holdout.')
    audit_sample: int = Field(description='Drawn into an open accuracy audit.')
    locked: int = Field(description='Already validated (by a person or an import).')
    deprecated_class: int = Field(description='Carrying a deprecated class or the region class.')


class AcceptMachineResponse(BaseModel):
    """``POST /label_ops/accept_machine``: what accepting would do (a dry run) or did.

    ``items`` and ``by_class`` count the labels the operation writes. A real run queues the
    recorded operation (``op_id`` and ``job_id``, both null when nothing qualified)."""

    dry_run: bool
    items: int = Field(description='Machine labels the operation accepts.')
    by_class: list[AcceptedClassCount] = Field(description='The items by class, most first.')
    held_back: AcceptHeldBack
    remaining: int = Field(
        description='Items beyond this operation (one writes at most 20,000): accept again.'
    )
    op_id: str | None = Field(default=None, description='The recorded operation (undo it by id).')
    job_id: str | None = Field(default=None, description='The queued job; poll GET /jobs/{job_id}.')


__all__ = [
    'AcceptHeldBack',
    'AcceptMachineResponse',
    'AcceptedClassCount',
    'AdoptDetectorRequest',
    'AdoptDetectorResponse',
    'CropUndoBatchResponse',
    'LabelOp',
    'LabelOpList',
    'LabelOpUndoPass',
    'LabelOpUndoRequest',
    'LabelOpUndoResponse',
]

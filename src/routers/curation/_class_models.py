"""Wire models for the class registry and new-class resolution routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.services.curation.class_sources import HumanLabelSource  # noqa: TC001


class ClassEntry(BaseModel):
    class_id: int
    class_name: str
    group: str = ''
    sample_count: int = 0
    validated_count: int = 0
    # FAISS-cluster bucket size: crops whose cluster_id == this class_id.
    # Includes unlabeled candidates that landed near the cluster — i.e.
    # everything visible on /clusters/{id}. The sidebar chip shows this
    # so the operator's eyes match what they'll see when they click in.
    cluster_size: int = 0
    deprecated: bool = False
    # Optional single-character keyboard shortcut. Persisted in the class
    # registry so user customizations survive across sessions and devices.
    # Validated server-side: must be one ASCII char, unique across active
    # classes, not collide with reserved shortcuts.
    hotkey_letter: str | None = None
    # ok / warn / block from validated_count (dataset_thresholds.py).
    adequacy: Literal['ok', 'warn', 'block'] = 'block'
    added_at: str | None = None


class ClassListResponse(BaseModel):
    classes: list[ClassEntry]
    thresholds: dict[str, int] = Field(default_factory=dict)
    # Single keys a class hotkey may not use (labeling actions).
    reserved_hotkeys: list[str] = Field(default_factory=list)


# Class names are slugs: they become export / training class names.
CLASS_NAME_PATTERN = r'^[a-z0-9_]+$'


class ClassCreateRequest(BaseModel):
    name: str = Field(pattern=CLASS_NAME_PATTERN)
    group: str = 'unknown'
    notes: str = ''
    # Optional; same rules as on update (one char, not reserved, unique).
    hotkey_letter: str | None = None


class ClassUpdateRequest(BaseModel):
    name: str | None = Field(default=None, pattern=CLASS_NAME_PATTERN)
    group: str | None = None
    # ``""`` clears the binding; ``None`` leaves it unchanged. Single ASCII
    # char only; uniqueness checked server-side at write time.
    hotkey_letter: str | None = None


class ClassMergeRequest(BaseModel):
    source_id: int
    target_id: int


class ResolveNewClassCreate(BaseModel):
    """``create`` payload on ``POST /review/new_class_proposals/resolve``:
    register a brand-new registry class before resolving the term. Same
    slug rule as ``ClassCreateRequest.name`` (``class_name`` here, to
    match the term the VLM proposed rather than an internal field name)."""

    class_name: str = Field(pattern=CLASS_NAME_PATTERN)
    group: str = 'unknown'
    notes: str | None = None


class ResolveNewClassRequest(BaseModel):
    """Bulk-resolve every pending ``vlm_new_class_pending`` item proposing
    ``label``. Exactly one of ``class_id`` (map to an existing registry
    class) / ``create`` (register a new one first) must be set."""

    label: str = Field(min_length=1)
    class_id: int | None = None
    create: ResolveNewClassCreate | None = None
    label_source: HumanLabelSource = 'new_class_proposal'


class ResolveConflict(BaseModel):
    crop_id: str
    current_source: str | None = None


class ResolveNewClassResponse(BaseModel):
    class_id: int | None
    class_name: str
    created: bool
    label: str
    matched: int
    matched_ids: list[str] = Field(default_factory=list)
    updated: int
    updated_ids: list[str] = Field(default_factory=list)
    conflicts: list[ResolveConflict] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)

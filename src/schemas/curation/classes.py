"""Wire shapes of the class-registry routes that answered with plain dicts."""

from __future__ import annotations

from pydantic import BaseModel


class ClassSourceEntry(BaseModel):
    id: str
    label: str
    role: str
    short_label: str


class ClassSourceList(BaseModel):
    class_sources: list[ClassSourceEntry]


class ClassCreated(BaseModel):
    """``POST /classes``: the class that was added."""

    class_id: int
    class_name: str
    group: str
    hotkey_letter: str | None = None


class ClassMergeDryRun(BaseModel):
    """``POST /classes/merge?dry_run=true``: what the merge would touch."""

    dry_run: bool
    source_id: int
    target_id: int
    would_relabel: int
    validations_carried_over: int
    holdout_blocking: int
    blocked: bool

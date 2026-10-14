"""Wire models for the projects lifecycle surface (§4). P1 only serves
``GET /projects`` and ``GET /projects/{project}``; the rest of this
module's shapes exist because §4's exact JSON needs them, not because
P1 implements their routes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config.projects import (
    CLONEABLE_AXES,
    PROJECT_SLUG_MAX_LEN,
    PROJECT_SLUG_MIN_LEN,
    PROJECT_SLUG_RE,
    RESERVED_SLUGS,
)
from src.schemas.curation.config_common import JobRefWire, ProjectCapacityWire


_STATUS_LABELS: dict[str, str] = {
    'active': 'Active',
    'archived': 'Archived',
    'building': 'Building',
    'failed': 'Failed',
    'deleting': 'Deleting',
    'deleted': 'Deleted',
}


class ProjectCounts(BaseModel):
    # ``null`` for images, items, validated and items_embedded when the count
    # could not be read (an outage), never a made-up 0.
    images: int | None = 0
    items: int | None = 0
    # Items with a human-validated class (the same count on every route,
    # ``src.services.projects.stats.validated_count``); ``null`` when it
    # could not be counted, never a made-up 0.
    validated: int | None = None
    # Items that have a vector (clustering, search and the VLM work on these);
    # the rest are stored but not embedded. ``null`` when uncountable.
    items_embedded: int | None = None


class ProjectSummary(BaseModel):
    slug: str
    display_name: str
    description: str
    prefix: str
    status: Literal['building', 'active', 'archived', 'deleting', 'deleted', 'failed']
    writable: bool
    selectable: bool
    is_default: bool
    deletable: bool
    # Whether POST .../archive / .../unarchive accepts this status; the
    # server refuses any other transition with 409 invalid_transition.
    archivable: bool
    unarchivable: bool
    revision: int
    created_at: str
    updated_at: str
    counts: ProjectCounts
    origin: dict[str, Any] | None = None
    # Cropwright BA-P2-4: the per-project pause flag, read straight off
    # this record's own state dir so GET /projects needs no per-row
    # GET .../pause call to render a "paused" chip.
    paused: bool = False


class ProjectLimits(BaseModel):
    slug_pattern: str = PROJECT_SLUG_RE
    slug_min: int = PROJECT_SLUG_MIN_LEN
    slug_max: int = PROJECT_SLUG_MAX_LEN
    reserved_slugs: list[str] = sorted(RESERVED_SLUGS)
    # Slugs of deleted projects: retired forever (a create answers 409
    # ``slug_retired``), served so a create form can flag them before submit.
    retired_slugs: list[str] = Field(default_factory=list)
    cloneable_axes: list[str] = list(CLONEABLE_AXES)


class ProjectLabels(BaseModel):
    """Display copy for served enums, so a client renders a status
    without its own table."""

    status: dict[str, str] = Field(default_factory=lambda: dict(_STATUS_LABELS))


class ProjectsResponse(BaseModel):
    default_slug: str
    projects: list[ProjectSummary]
    capacity: ProjectCapacityWire | None
    limits: ProjectLimits
    labels: ProjectLabels = Field(default_factory=ProjectLabels)
    include_archived: bool


class ProjectWarning(BaseModel):
    """A non-blocking note on a lifecycle response (e.g.
    ``shard_budget_high``)."""

    code: str
    message: str
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description='Structured facts behind the note. For shard_budget_high: the same '
        'capacity block GET /curation/projects serves.',
    )


class KeymapCloneConflictWire(BaseModel):
    """One action whose combo collides with the target's class hotkey.
    The ``keymap`` clone axis is all-or-nothing: on any conflict the whole
    keymap is skipped and the target's keymap is left unchanged -- a
    report, never a silent unbind (CW-K §0 clause 1)."""

    action_id: str
    combo: str
    class_id: int
    class_name: str


class ProjectLifecycleResponse(BaseModel):
    """Every project lifecycle mutation (create 201, PATCH, archive,
    unarchive, clone_settings; P3) answers this envelope, so the switcher
    adopts the returned summary without a re-read."""

    project: ProjectSummary
    warnings: list[ProjectWarning] = Field(default_factory=list)
    keymap_clone_conflicts: list[KeymapCloneConflictWire] = Field(default_factory=list)


class ProjectError(BaseModel):
    """Why a ``failed`` project failed."""

    code: str
    message: str


class ProjectRecordResponse(ProjectSummary):
    """``GET {prefix}`` (``/projects/{project}``): exactly ``ProjectSummary``
    + ``resources`` (paths as served strings) + ``error`` (null unless
    ``status == "failed"``, which P1 never produces -- create is P3
    scope)."""

    resources: dict[str, Any]
    error: ProjectError | None = None


class CreateProjectRequest(BaseModel):
    """``POST /projects`` (P3)."""

    slug: str
    display_name: str
    description: str = ''
    clone_settings_from: str | None = None
    clone_axes: list[str] | None = None


class PatchProjectRequest(BaseModel):
    """``PATCH {prefix}`` (P3). The slug is immutable."""

    display_name: str | None = None
    description: str | None = None
    expected_revision: int


class ArchiveRequest(BaseModel):
    expected_revision: int


class CloneSettingsRequest(BaseModel):
    """``POST {prefix}/clone_settings`` (P3)."""

    from_: str = Field(alias='from')
    axes: list[str] | None = None
    expected_revision: int

    model_config = {'populate_by_name': True}


class DeleteBlockingIssue(BaseModel):
    """One reason a delete is refused (delta 11: structured, not bare
    codes)."""

    code: str
    message: str


class DryRunIndexReport(BaseModel):
    name: str
    # P3F m10: null when the count genuinely could not be taken, matching
    # ProjectCounts.validated's null-not-0 rule -- never a made-up 0.
    docs: int | None = None
    store_bytes: int | None = None


class DryRunDirReport(BaseModel):
    path: str
    bytes: int


class DeleteDryRunResponse(BaseModel):
    indexes: list[DryRunIndexReport]
    dirs: list[DryRunDirReport]
    promoted_models: list[str] = Field(default_factory=list)
    mlflow_experiment: str
    # P3F m5: typed, not a raw dict -- same JobRefWire every 409
    # project_busy carries.
    running_jobs: list[JobRefWire] = Field(default_factory=list)
    referenced_by: list[dict[str, Any]] = Field(default_factory=list)
    blocking: list[str]
    blocking_detail: list[DeleteBlockingIssue]


class ProjectStatsCounts(BaseModel):
    # Every OpenSearch count is ``null`` when it could not be read.
    images: int | None
    items: int | None
    validated: int | None
    items_embedded: int | None
    pending_detection: int | None
    holdout_items: int | None
    classes: int
    promoted_models: int


class ProjectStatsResponse(BaseModel):
    counts: ProjectStatsCounts
    indexes: list[dict[str, Any]]
    disk: dict[str, Any]
    jobs: dict[str, Any]
    last_ingest_at: str | None = None

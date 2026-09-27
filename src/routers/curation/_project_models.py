"""Wire models for the projects lifecycle surface (§4). P1 only serves
``GET /projects`` and ``GET /projects/{project}``; the rest of this
module's shapes exist because §4's exact JSON needs them, not because
P1 implements their routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.config.projects import (
    PROJECT_SLUG_MAX_LEN,
    PROJECT_SLUG_MIN_LEN,
    PROJECT_SLUG_RE,
    RESERVED_SLUGS,
)
from src.routers.curation._config_common_models import ProjectCapacityWire


if TYPE_CHECKING:
    from src.services.projects.capacity import ProjectCapacity


_STATUS_LABELS: dict[str, str] = {
    'active': 'Active',
    'archived': 'Archived',
    'building': 'Building',
    'failed': 'Failed',
    'deleting': 'Deleting',
    'deleted': 'Deleted',
}

# List membership by status (§4 rev 2 / delta 3): active always;
# archived only opt-in; building/failed/deleting always (so a stuck
# create or an in-flight combine target survive a reload); deleted
# never.
_ALWAYS_LISTED_STATUSES = frozenset({'active', 'building', 'failed', 'deleting'})
_ARCHIVED_STATUS = 'archived'

CLONEABLE_AXES: tuple[str, ...] = ('settings_defaults', 'classes', 'activations')

# The only statuses archive / unarchive act on (lifecycle.py enforces them).
ARCHIVABLE_STATUSES = frozenset({'active'})
UNARCHIVABLE_STATUSES = frozenset({'archived'})


class ProjectCounts(BaseModel):
    images: int = 0
    items: int = 0
    # Items with a human-validated class (the same count on every route,
    # ``src.services.projects.stats.validated_count``); ``null`` when it
    # could not be counted, never a made-up 0.
    validated: int | None = None


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


def list_membership(status: str, *, include_archived: bool) -> bool:
    """Whether a project of this ``status`` appears in ``GET /projects``'s
    ``projects[]`` (§4 delta 3)."""
    if status == _ARCHIVED_STATUS:
        return include_archived
    return status in _ALWAYS_LISTED_STATUSES


def summarize(record: Any, counts: ProjectCounts) -> ProjectSummary:
    from src.config.project_context import bind_project
    from src.config.projects import DEFAULT_SLUG

    with bind_project(record):
        from src.config.project_context import project_api_base

        prefix = project_api_base()
    is_default = record.slug == DEFAULT_SLUG
    return ProjectSummary(
        slug=record.slug,
        display_name=record.display_name,
        description=record.description,
        prefix=prefix,
        status=record.status,
        writable=record.status == 'active',
        selectable=record.status in ('active', 'archived'),
        is_default=is_default,
        deletable=not is_default,
        archivable=record.status in ARCHIVABLE_STATUSES,
        unarchivable=record.status in UNARCHIVABLE_STATUSES,
        revision=record.revision,
        created_at=record.created_at,
        updated_at=record.updated_at,
        counts=counts,
        origin=record.origin,
    )


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


class ProjectLifecycleResponse(BaseModel):
    """Every project lifecycle mutation (create 201, PATCH, archive,
    unarchive, clone_settings; P3) answers this envelope, so the switcher
    adopts the returned summary without a re-read."""

    project: ProjectSummary
    warnings: list[ProjectWarning] = Field(default_factory=list)


class ProjectError(BaseModel):
    """Why a ``failed`` project failed."""

    code: str
    message: str


def capacity_wire(capacity: ProjectCapacity | None) -> ProjectCapacityWire | None:
    return ProjectCapacityWire(**capacity.to_wire()) if capacity is not None else None


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
    docs: int
    store_bytes: int | None = None


class DryRunDirReport(BaseModel):
    path: str
    bytes: int


class DeleteDryRunResponse(BaseModel):
    indexes: list[DryRunIndexReport]
    dirs: list[DryRunDirReport]
    promoted_models: list[str] = Field(default_factory=list)
    mlflow_experiment: str
    running_jobs: list[dict[str, Any]] = Field(default_factory=list)
    referenced_by: list[dict[str, Any]] = Field(default_factory=list)
    blocking: list[str]
    blocking_detail: list[DeleteBlockingIssue]


class ProjectStatsCounts(BaseModel):
    images: int
    items: int
    # Same count and same null-when-uncountable rule as ProjectCounts.validated.
    validated: int | None
    pending_detection: int
    holdout_items: int
    classes: int
    promoted_models: int


class ProjectStatsResponse(BaseModel):
    counts: ProjectStatsCounts
    indexes: list[dict[str, Any]]
    disk: dict[str, Any]
    jobs: dict[str, Any]
    last_ingest_at: str | None = None


def resources_wire(resources: Any) -> dict[str, Any]:
    return {
        'indexes': {role.value: name for role, name in resources.indexes.items()},
        'class_registry_path': str(resources.class_registry_path),
        'export_root': str(resources.export_root),
        'upload_root': str(resources.upload_root),
        'bakeoff_eval_root': str(resources.bakeoff_eval_root),
        'project_state_dir': str(resources.project_state_dir),
        'train_jobs_dir': str(resources.train_jobs_dir),
        'autolabel_dir': str(resources.autolabel_dir),
        'bakeoff_jobs_dir': str(resources.bakeoff_jobs_dir),
        'mlflow_experiment': resources.mlflow_experiment,
        'model_prefix': resources.model_prefix,
    }

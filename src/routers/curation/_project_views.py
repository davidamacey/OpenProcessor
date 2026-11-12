"""Record-to-wire helpers of the projects routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.projects import ARCHIVABLE_STATUSES, UNARCHIVABLE_STATUSES
from src.schemas.curation.config_common import ProjectCapacityWire
from src.schemas.curation.projects import ProjectCounts, ProjectSummary
from src.services.projects.listing import SELECTABLE


if TYPE_CHECKING:
    from src.services.projects.capacity import ProjectCapacity


def summarize(record: Any, counts: ProjectCounts) -> ProjectSummary:
    from src.config.project_context import bind_project
    from src.config.projects import DEFAULT_SLUG

    with bind_project(record):
        from src.config.project_context import project_api_base

        prefix = project_api_base()
    is_default = record.slug == DEFAULT_SLUG
    paused = record.paused_at is not None
    return ProjectSummary(
        slug=record.slug,
        display_name=record.display_name,
        description=record.description,
        prefix=prefix,
        status=record.status,
        writable=record.status == 'active',
        selectable=record.status in SELECTABLE,
        is_default=is_default,
        deletable=not is_default,
        archivable=record.status in ARCHIVABLE_STATUSES,
        unarchivable=record.status in UNARCHIVABLE_STATUSES,
        revision=record.revision,
        created_at=record.created_at,
        updated_at=record.updated_at,
        counts=counts,
        origin=record.origin,
        paused=paused,
    )


def capacity_wire(capacity: ProjectCapacity | None) -> ProjectCapacityWire | None:
    return ProjectCapacityWire(**capacity.to_wire()) if capacity is not None else None


def resources_wire(resources: Any) -> dict[str, Any]:
    return {
        'indexes': {role.value: name for role, name in resources.indexes.items()},
        'class_registry_path': str(resources.class_registry_path),
        'export_root': str(resources.export_root),
        'upload_root': str(resources.upload_root),
        'bakeoff_eval_root': str(resources.bakeoff_eval_root),
        'project_state_dir': str(resources.project_state_dir),
        'bakeoff_results_dir': str(resources.bakeoff_results_dir),
        'mlflow_experiment': resources.mlflow_experiment,
        'model_prefix': resources.model_prefix,
    }

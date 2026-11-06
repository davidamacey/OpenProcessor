"""The two conversions between a stored project and the rest of the code."""

from __future__ import annotations

from typing import Any

from src.config.curation_accessors import base_curation_config
from src.config.projects import ProjectRecord, new_project_names, resources_from_stored
from src.db.repos.projects import NewProject, ProjectRow


def record_from_row(row: ProjectRow) -> ProjectRecord:
    """The wire-facing record of a stored project."""
    return ProjectRecord(
        id=row.id,
        slug=row.slug,
        display_name=row.display_name,
        description=row.description,
        status=row.status,  # type: ignore[arg-type]  # the table CHECK is the closed set
        revision=row.revision,
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
        origin=row.origin,
        resources=resources_from_stored(
            row.slug,
            base_curation_config(),
            index_prefix=row.index_prefix,
            model_prefix=row.model_prefix,
            mlflow_experiment=row.mlflow_experiment,
            indexes=row.indexes,
        ),
        pre_delete_status=row.pre_delete_status,
        paused_at=row.paused_at.isoformat() if row.paused_at is not None else None,
        paused_by=row.paused_by,
        storage_prefix=row.storage_prefix,
    )


def new_project_for(
    slug: str,
    *,
    display_name: str,
    description: str = '',
    origin: Any | None = None,
    status: str = 'building',
    created_by: str = 'api',
) -> NewProject:
    """What a create writes for ``slug``: the names are computed here once and stored."""
    names = new_project_names(slug)
    return NewProject(
        slug=slug,
        display_name=display_name,
        description=description,
        status=status,
        origin=origin,
        index_prefix=names.index_prefix,
        storage_prefix=names.storage_prefix,
        model_prefix=names.model_prefix,
        mlflow_experiment=names.mlflow_experiment,
        indexes=names.indexes,
        created_by=created_by,
    )

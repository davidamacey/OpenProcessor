"""Project resource naming and records (see
``docs/design/openprocessor_internal/projects_plan.md`` §2.2).

A *project* is a named, isolated dataset workspace. ``default`` is an
ordinary project, created at bootstrap with the same naming as every
other one (:func:`resources_for_new`); it is only protected (archive, never
delete; owner decision D5).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.config.curation import CurationConfig, IndexRole


# 2..32 chars, lowercase letters/digits, single hyphens between segments.
# No underscore (the project-index separator ``__`` must be unambiguous)
# and no leading digit/hyphen.
PROJECT_SLUG_RE = r'^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$'
PROJECT_SLUG_MIN_LEN = 2
PROJECT_SLUG_MAX_LEN = 32

# Reserved because they collide with sibling route segments or would be
# ambiguous as a slug (``combine`` is a job endpoint under
# ``/projects/combine/...``, not a project).
RESERVED_SLUGS = frozenset(
    {
        'combine',
        'new',
        'all',
        'none',
        'projects',
        'global',
        'settings',
        'vlm',
        'health',
    }
)

DEFAULT_SLUG = 'default'

_SLUG_PATTERN = re.compile(PROJECT_SLUG_RE)


def is_valid_slug(slug: str) -> bool:
    """Full slug validity: pattern, length, not reserved."""
    if not (PROJECT_SLUG_MIN_LEN <= len(slug) <= PROJECT_SLUG_MAX_LEN):
        return False
    if slug in RESERVED_SLUGS:
        return False
    return bool(_SLUG_PATTERN.match(slug))


@dataclass(frozen=True)
class ProjectResources:
    """Every resource a bound project needs, resolved once at create time
    (bootstrap, for ``default``) and then treated as immutable — a later env change must never remap a live
    project's data."""

    indexes: Mapping[IndexRole, str]
    class_registry_path: Path
    export_root: Path
    upload_root: Path
    bakeoff_eval_root: Path
    project_state_dir: Path
    train_jobs_dir: Path
    autolabel_dir: Path
    bakeoff_jobs_dir: Path
    mlflow_experiment: str
    model_prefix: str


ProjectStatus = Literal['building', 'active', 'archived', 'deleting', 'deleted', 'failed']


@dataclass(frozen=True)
class ProjectRecord:
    """The persisted, wire-facing shape of a project (index ``op_projects``,
    doc id ``project:<slug>``)."""

    slug: str
    display_name: str
    description: str
    status: ProjectStatus
    revision: int
    created_at: str
    updated_at: str
    origin: dict | None
    resources: ProjectResources


def _project_index_prefix() -> str:
    return os.environ.get('OP_PROJECT_INDEX_PREFIX', 'op_prj_')


def projects_data_root() -> Path:
    return Path(os.environ.get('OP_PROJECTS_DATA_ROOT', './data/projects'))


def trainer_jobs_root() -> Path:
    """The shared trainer volume root (``OP_TRAIN_JOBS_DIR``). Each
    project's ``train_jobs_dir`` nests under ``<root>/projects/<slug>``;
    trainer-global files (``.trainer_capabilities.json``) live at the
    root itself, since one trainer serves every project."""
    return Path(os.environ.get('OP_TRAIN_JOBS_DIR', '/jobs'))


def resources_for_new(slug: str, base: CurationConfig) -> ProjectResources:
    """Resources for a brand-new project ``slug``, per the §2.2 naming
    table. Computed once at create time and persisted -- never
    recomputed from a later env on every boot."""
    from src.config.curation import IndexRole

    prefix = _project_index_prefix()
    data_root = projects_data_root() / slug
    indexes = {role: f'{prefix}{slug}__{role.value}' for role in IndexRole}
    return ProjectResources(
        indexes=indexes,
        class_registry_path=data_root / 'class_registry.json',
        export_root=data_root / 'exports',
        upload_root=base.state_dir / 'projects' / slug / 'uploads',
        bakeoff_eval_root=data_root / 'bakeoff_eval',
        project_state_dir=base.state_dir / 'projects' / slug,
        train_jobs_dir=trainer_jobs_root() / 'projects' / slug,
        autolabel_dir=Path(os.environ.get('OP_AUTO_LABEL_STATE_DIR', '/jobs/auto_label'))
        / 'projects'
        / slug,
        bakeoff_jobs_dir=base.state_dir / 'projects' / slug / 'bakeoff_jobs',
        mlflow_experiment=f'openprocessor-{slug}',
        model_prefix=f'{slug}__',
    )


def new_project_record(
    slug: str,
    base: CurationConfig,
    *,
    display_name: str | None = None,
    description: str = '',
    status: ProjectStatus = 'active',
    now: str = '',
) -> ProjectRecord:
    """A fresh record for ``slug`` with :func:`resources_for_new` -- the one
    way every project, ``default`` included, gets its resources."""
    return ProjectRecord(
        slug=slug,
        display_name=display_name or slug.replace('-', ' ').title(),
        description=description,
        status=status,
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base),
    )

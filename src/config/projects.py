"""Project resource naming and records (see
``docs/design/openprocessor_internal/projects_plan.md`` §2.2).

A *project* is a named, isolated dataset workspace. ``default`` is an
ordinary project whose resources happen to resolve to today's env-driven
index names and paths, so nothing that already runs against the default
project needs to change.
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
    """Every resource a bound project needs, resolved once (at default-boot
    time for ``default``, at create time for a new project) and then
    treated as immutable — a later env change must never remap a live
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


def _projects_data_root() -> Path:
    return Path(os.environ.get('OP_PROJECTS_DATA_ROOT', './data/projects'))


def resources_for_default(base: CurationConfig) -> ProjectResources:
    """Today's env-derived values, computed fresh at every boot -- the
    ``default`` project is not a special case, it is the project whose
    resources happen to already exist."""
    from src.config.curation import IndexRole, index_name

    indexes = {role: index_name(base, role) for role in IndexRole}
    return ProjectResources(
        indexes=indexes,
        class_registry_path=base.class_registry_path,
        export_root=base.export_root,
        upload_root=base.upload_root,
        bakeoff_eval_root=base.bakeoff_eval_root,
        project_state_dir=base.state_dir,
        train_jobs_dir=Path(os.environ.get('OP_TRAIN_JOBS_DIR', '/jobs')),
        autolabel_dir=Path(os.environ.get('OP_AUTO_LABEL_STATE_DIR', '/jobs/auto_label')),
        bakeoff_jobs_dir=Path(
            os.environ.get('OP_BAKEOFF_JOBS_DIR', str(base.state_dir / 'bakeoff_jobs'))
        ),
        mlflow_experiment=os.environ.get('MLFLOW_EXPERIMENT_NAME', 'openprocessor'),
        model_prefix='',
    )


def resources_for_new(slug: str, base: CurationConfig) -> ProjectResources:
    """Resources for a brand-new project ``slug``, per the §2.2 naming
    table. Computed once at create time and persisted -- never
    recomputed from a later env on every boot."""
    from src.config.curation import IndexRole

    prefix = _project_index_prefix()
    data_root = _projects_data_root() / slug
    indexes = {role: f'{prefix}{slug}__{role.value}' for role in IndexRole}
    # Shard folding (owner D4, projects_plan.md §2.3): a project created
    # after W2 never gets its own settings / umap-viz-state indexes --
    # both roles resolve to the configs index name instead (6 indexes,
    # not 8). ``default`` is unaffected (resources_for_default keeps the
    # env-derived names). Safe because both folded roles are addressed
    # by a single fixed doc id only (never searched) and their mappings
    # merge without a type conflict -- see _configs_body().
    configs_name = f'{prefix}{slug}__{IndexRole.CONFIGS.value}'
    indexes[IndexRole.SETTINGS] = configs_name
    indexes[IndexRole.UMAP_VIZ_STATE] = configs_name
    return ProjectResources(
        indexes=indexes,
        class_registry_path=data_root / 'class_registry.json',
        export_root=data_root / 'exports',
        upload_root=base.state_dir / 'projects' / slug / 'uploads',
        bakeoff_eval_root=data_root / 'bakeoff_eval',
        project_state_dir=base.state_dir / 'projects' / slug,
        train_jobs_dir=Path(os.environ.get('OP_TRAIN_JOBS_DIR', '/jobs')) / 'projects' / slug,
        autolabel_dir=Path(os.environ.get('OP_AUTO_LABEL_STATE_DIR', '/jobs/auto_label'))
        / 'projects'
        / slug,
        bakeoff_jobs_dir=base.state_dir / 'projects' / slug / 'bakeoff_jobs',
        mlflow_experiment=f'openprocessor-{slug}',
        model_prefix=f'{slug}__',
    )

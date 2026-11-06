"""Project resource naming and records (see
the project-isolation design §2.2).

A *project* is a named, isolated dataset workspace. ``default`` is an
ordinary project, created at bootstrap with the same naming as every
other one (:func:`resources_for_new`); it is only protected (archive, never
delete; owner decision D5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.config.env import env_path, env_str


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from pathlib import Path

    from src.config.curation import CurationConfig
    from src.config.curation_roles import IndexRole


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
    """The wire-facing shape of a project, built from its ``projects`` row."""

    slug: str
    display_name: str
    description: str
    status: ProjectStatus
    revision: int
    created_at: str
    updated_at: str
    origin: dict | None
    resources: ProjectResources
    # The status a 'deleting' record rolls back to if the drain wait times out;
    # None when the project is not mid-delete.
    pre_delete_status: str | None = None
    # The database row's identity (what ``project_tx`` scopes to). None only for a
    # record built without a database behind it.
    id: uuid.UUID | None = None
    paused_at: str | None = None
    paused_by: str | None = None
    # Relative key prefix of the project's files behind the storage seam ('projects/<slug>').
    storage_prefix: str = ''


def project_index_prefix() -> str:
    return env_str('OP_PROJECT_INDEX_PREFIX', 'op_prj_')


def projects_data_root() -> Path:
    return env_path('OP_PROJECTS_DATA_ROOT', './data/projects')


def trainer_jobs_root() -> Path:
    """The shared trainer volume root (``OP_TRAIN_JOBS_DIR``). Each
    project's ``train_jobs_dir`` nests under ``<root>/projects/<slug>``;
    trainer-global files (``.trainer_capabilities.json``) live at the
    root itself, since one trainer serves every project."""
    return env_path('OP_TRAIN_JOBS_DIR', '/jobs')


# Index roles that still exist in OpenSearch although ``project_indexes`` records only ``images``
# and ``items``. Their names follow the creation rule; lanes 2B-2D delete the last users and this
# set with them.
_DERIVED_INDEX_ROLES = (
    'labels_confirmed',
    'classes',
    'umap_state',
    'configs',
)
_FOLDED_ON_CONFIGS = ('settings', 'umap_viz_state')


def resources_from_stored(
    slug: str,
    base: CurationConfig,
    *,
    index_prefix: str,
    model_prefix: str,
    mlflow_experiment: str,
    indexes: Mapping[str, str],
) -> ProjectResources:
    """A project's resources from what its row stores (index names, prefixes, experiment);
    the directories derive from the slug and the deployment's roots."""
    from src.config.curation_roles import IndexRole

    by_role: dict[IndexRole, str] = {IndexRole(role): name for role, name in indexes.items()}
    for role in _DERIVED_INDEX_ROLES:
        by_role[IndexRole(role)] = f'{index_prefix}{slug}__{role}'
    # Shard folding (owner D4): settings and umap-viz-state live on the configs index.
    configs_name = by_role[IndexRole.CONFIGS]
    for role in _FOLDED_ON_CONFIGS:
        by_role[IndexRole(role)] = configs_name
    data_root = projects_data_root() / slug
    return ProjectResources(
        indexes=by_role,
        class_registry_path=data_root / 'class_registry.json',
        export_root=data_root / 'exports',
        upload_root=base.state_dir / 'projects' / slug / 'uploads',
        bakeoff_eval_root=data_root / 'bakeoff_eval',
        project_state_dir=base.state_dir / 'projects' / slug,
        train_jobs_dir=trainer_jobs_root() / 'projects' / slug,
        autolabel_dir=env_path('OP_AUTO_LABEL_STATE_DIR', '/jobs/auto_label') / 'projects' / slug,
        bakeoff_jobs_dir=base.state_dir / 'projects' / slug / 'bakeoff_jobs',
        mlflow_experiment=mlflow_experiment,
        model_prefix=model_prefix,
    )


@dataclass(frozen=True)
class ProjectNames:
    """The names a project is created with; stored on its row, never recomputed."""

    index_prefix: str
    storage_prefix: str
    model_prefix: str
    mlflow_experiment: str
    indexes: Mapping[str, str]


def new_project_names(slug: str) -> ProjectNames:
    """The names a brand-new project ``slug`` is created with, per the naming table.

    ``default``'s model prefix stays empty (not ``default__``) so every model promoted before
    projects existed, and every core pipeline model, keeps resolving as ``default``'s own.
    """
    prefix = project_index_prefix()
    return ProjectNames(
        index_prefix=prefix,
        storage_prefix=f'projects/{slug}',
        model_prefix='' if slug == DEFAULT_SLUG else f'{slug}__',
        mlflow_experiment=f'openprocessor-{slug}',
        indexes={role: f'{prefix}{slug}__{role}' for role in ('images', 'items')},
    )


def resources_for_new(slug: str, base: CurationConfig) -> ProjectResources:
    """Resources for a brand-new project ``slug``."""
    names = new_project_names(slug)
    return resources_from_stored(
        slug,
        base,
        index_prefix=names.index_prefix,
        model_prefix=names.model_prefix,
        mlflow_experiment=names.mlflow_experiment,
        indexes=names.indexes,
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


CLONEABLE_AXES: tuple[str, ...] = (
    'settings_defaults',
    'classes',
    'activations',
    'keymap',
    'prompt_packs',
    'open_vocab',
    'ingest_profile',
    'vlm_activation',
)

# The only statuses archive / unarchive act on (lifecycle.py enforces them).
ARCHIVABLE_STATUSES = frozenset({'active'})

UNARCHIVABLE_STATUSES = frozenset({'archived'})

"""Accessors for the bound project's curation configuration.

``get_curation_config()`` answers global fields from the env-built base
instance and project-scoped fields from the project bound in the current
context; ``idx`` and the ``*_index()`` functions resolve the bound
project's index names at call time.
"""

from __future__ import annotations

from typing import Any, cast

from src.config.curation import CurationConfig
from src.config.curation_roles import INDEX_ROLE_ATTR, IndexRole, index_name


# Fields resolved from the *bound project*'s ``ProjectResources`` rather
# than the process-global env-built instance (see
# the project-isolation design §2.2/§3.3). Every
# other ``CurationConfig`` field is global. ``test_config_view.py``
# fails if a new dataclass field is added to neither this set nor
# treated as global -- keep it in sync with the dataclass above.
#
PROJECT_SCOPED_FIELDS: frozenset[str] = frozenset(
    {
        'images_index',
        'items_index',
        'labels_confirmed_index',
        'classes_index',
        'settings_index',
        'umap_state_index',
        'umap_viz_state_index',
        'configs_index',
        'class_registry_path',
        'export_root',
        'upload_root',
        'bakeoff_eval_root',
        'project_slug',
        'project_state_dir',
        'train_jobs_dir',
        'autolabel_dir',
        'bakeoff_jobs_dir',
        'mlflow_experiment',
        'model_prefix',
    }
)

# Maps a PROJECT_SCOPED_FIELDS index-role field name to the IndexRole its
# value comes from on the bound project's ``ProjectResources.indexes``.
_PROJECT_FIELD_INDEX_ROLE: dict[str, IndexRole] = dict(
    zip(INDEX_ROLE_ATTR.values(), INDEX_ROLE_ATTR.keys(), strict=True)
)
# ^ inverts {IndexRole: attr_name} -> {attr_name: IndexRole}; both sides
# of ``INDEX_ROLE_ATTR`` are unique so this round-trips exactly.

# Maps a PROJECT_SCOPED_FIELDS non-index field name to the matching
# attribute on ``ProjectResources``.
_PROJECT_FIELD_RESOURCE_ATTR: dict[str, str] = {
    'class_registry_path': 'class_registry_path',
    'export_root': 'export_root',
    'upload_root': 'upload_root',
    'bakeoff_eval_root': 'bakeoff_eval_root',
    'project_state_dir': 'project_state_dir',
    'train_jobs_dir': 'train_jobs_dir',
    'autolabel_dir': 'autolabel_dir',
    'bakeoff_jobs_dir': 'bakeoff_jobs_dir',
    'mlflow_experiment': 'mlflow_experiment',
    'model_prefix': 'model_prefix',
}


class CurationConfigView:
    """A ``CurationConfig``-shaped view that resolves PROJECT_SCOPED_FIELDS
    from :func:`~src.config.project_context.current_project` at attribute
    access time, and every other field from the process-global base
    instance.

    Because resolution happens on each ``getattr``, the 15+ existing
    ``config = get_curation_config()`` module-level captures need no
    edits: a later ``bind_project`` call is picked up by the very next
    attribute access on that same captured object.

    Fail closed (§0 principle 3): reading a project-scoped field with no
    project bound raises :class:`~src.config.project_context.ProjectNotBound`;
    it never falls back to ``default``. Global fields work unbound.
    """

    __slots__ = ('_base',)

    def __init__(self, base: CurationConfig) -> None:
        self._base = base

    def __getattr__(self, name: str) -> Any:
        if name not in PROJECT_SCOPED_FIELDS:
            return getattr(self._base, name)
        from src.config.project_context import current_project

        bound = current_project()
        resources = bound.record.resources
        if name == 'project_slug':
            return bound.record.slug
        if name in _PROJECT_FIELD_INDEX_ROLE:
            return resources.indexes[_PROJECT_FIELD_INDEX_ROLE[name]]
        return getattr(resources, _PROJECT_FIELD_RESOURCE_ATTR[name])

    def __repr__(self) -> str:
        return f'CurationConfigView(base={self._base!r})'


_default_curation_config: CurationConfig | None = None


def base_curation_config() -> CurationConfig:
    """The process-global, env-built ``CurationConfig`` instance -- the
    global-field half of :func:`get_curation_config`'s view, and the
    right thing to pass to ``resources_for_new`` (which only reads global
    fields).

    Built via :meth:`CurationConfig.from_env` so the ``OP_*`` env vars
    documented on that classmethod (e.g. ``OP_API_PREFIX``) actually take
    effect — this was previously constructing a bare ``CurationConfig()``
    and silently ignoring every ``OP_*`` override.
    """
    global _default_curation_config  # noqa: PLW0603 - lazily-built module singleton
    if _default_curation_config is None:
        _default_curation_config = CurationConfig.from_env()
    return _default_curation_config


def get_curation_config() -> CurationConfig:
    """The process-wide curation config: global fields from the env-built
    base instance, project-scoped fields (PROJECT_SCOPED_FIELDS) from
    whatever project is bound in the current context. Typed as
    ``CurationConfig`` (it is a ``CurationConfig``-shaped view, not a
    subclass) so every existing typed call site needs no edits.
    """
    return cast('CurationConfig', CurationConfigView(base_curation_config()))


def idx(role: IndexRole) -> str:
    """``index_name(get_curation_config(), role)`` -- the bound project's
    index name for ``role``."""
    return index_name(get_curation_config(), role)


# The bound project's index name per role, resolved at call time. Use
# these (never a module-level constant): a name captured at import would
# pin every request to one project's index.
def images_index() -> str:
    return idx(IndexRole.IMAGES)


def items_index() -> str:
    return idx(IndexRole.ITEMS)


def labels_confirmed_index() -> str:
    return idx(IndexRole.LABELS_CONFIRMED)


def classes_index() -> str:
    return idx(IndexRole.CLASSES)


def umap_state_index() -> str:
    return idx(IndexRole.UMAP_STATE)


def umap_viz_state_index() -> str:
    return idx(IndexRole.UMAP_VIZ_STATE)

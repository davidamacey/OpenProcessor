"""Logical roles of the curation OpenSearch indexes.

``IndexRole`` names a stable role; :func:`index_name` resolves the
deployment's configured index name for a role from a ``CurationConfig``.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.config.curation import CurationConfig


class IndexRole(str, Enum):  # noqa: UP042  # (str, Enum) keeps str() and format() as Class.MEMBER; StrEnum would change both
    """Logical role of a curation OpenSearch index.

    Unlike the reference private string enum, member *values* are
    stable role identifiers, not deployment-specific index names — the
    actual index name for a role is resolved via :func:`index_name`
    against a :class:`CurationConfig` instance, so it can be
    overridden per deployment without touching this enum.
    """

    IMAGES = 'images'
    ITEMS = 'items'
    LABELS_CONFIRMED = 'labels_confirmed'
    CLASSES = 'classes'
    SETTINGS = 'settings'
    # The two UMAP state indexes (see the ``umap_state_index`` /
    # ``umap_viz_state_index`` fields below) were previously auto-created
    # by dynamic mapping on first ``client.index()`` call -- no explicit
    # mapping, replicas=1 (keeps a single-node cluster yellow). They now
    # go through the same INDEX_BODIES bootstrap as every other curation
    # index.
    UMAP_STATE = 'umap_state'
    UMAP_VIZ_STATE = 'umap_viz_state'
    # The config store: prompt packs, region profiles, activations and the
    # cross-process revision counter. For a new project,
    # ``resources_for_new`` also folds SETTINGS and UMAP_VIZ_STATE onto
    # this index's name (projects_plan.md §2.3, owner D4) -- ``default``
    # keeps its own separate settings/umap-viz-state indexes.
    CONFIGS = 'configs'


INDEX_ROLE_ATTR: dict[IndexRole, str] = {
    IndexRole.IMAGES: 'images_index',
    IndexRole.ITEMS: 'items_index',
    IndexRole.LABELS_CONFIRMED: 'labels_confirmed_index',
    IndexRole.CLASSES: 'classes_index',
    IndexRole.SETTINGS: 'settings_index',
    IndexRole.UMAP_STATE: 'umap_state_index',
    IndexRole.UMAP_VIZ_STATE: 'umap_viz_state_index',
    IndexRole.CONFIGS: 'configs_index',
}


def index_name(cfg: CurationConfig, role: IndexRole) -> str:
    """Resolve the configured OpenSearch index name for a logical role.

    Replaces reference call sites of the shape ``<PrivateIndexEnum>.X.value`` —
    those hardcoded a *value*; this looks the value up on ``cfg`` so it
    is deployment-overridable.
    """
    return getattr(cfg, INDEX_ROLE_ATTR[role])

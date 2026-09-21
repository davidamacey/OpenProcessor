"""Generic curation/labeling subsystem configuration.

Holds index names, filesystem roots and API-surface constants for the
`curation` namespace (OpenProcessor's generic port of a private
reference vehicle/license-plate curation stack — see
``docs/design/curation_design_rationale.md`` §2.1).

``CurationConfig`` replaces the module-level constants and the
``KbIndex`` string enum that the reference implementation hardcoded.
Index *values* are deployment data (a given operator's OpenSearch may
already have data under different index names), so they live on this
dataclass rather than in code. ``IndexRole`` + ``index_name()`` give a
stable, typo-proof way to look a configured index name up by its
logical role.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping


class IndexRole(str, Enum):
    """Logical role of a curation OpenSearch index.

    Unlike the reference ``KbIndex(str, Enum)``, member *values* are
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


@dataclass(frozen=True)
class CurationConfig:
    """Index names, filesystem roots and API-surface config for curation.

    Defaults are the generic OSS names. A deployment with existing data
    under different names (e.g. a proprietary-dataset overlay)
    constructs its own instance — see the design rationale in
    ``docs/design/curation_design_rationale.md`` §2.1. That overlay
    is not part of this generic module.
    """

    images_index: str = 'op_images'
    items_index: str = 'op_items'
    labels_confirmed_index: str = 'op_labels_confirmed'
    classes_index: str = 'op_classes'
    clusters_index: str = 'op_clusters'
    # Single shared-defaults document (curation-strategy settings) — one
    # doc, not a full index of many rows. See
    # ``src.clients.curation_opensearch.CURATION_SETTINGS_DOC_ID`` for the
    # fixed doc id this index always addresses.
    settings_index: str = 'op_curation_settings'
    # Two deliberately distinct UMAP-state indexes (see
    # ``src/services/curation/embedding_viz.py`` module docstring):
    # the retired clustering reducer's fitted-manifold cache
    # (``clustering/embedding_reduce.py``) and the visualization-only
    # projection's own metadata slot. They must never share a name or
    # state, so they get separate fields rather than one shared role.
    umap_state_index: str = 'op_umap_state'
    umap_viz_state_index: str = 'op_umap_viz_state'

    class_registry_path: Path = Path('./data/class_registry.json')
    # Optional deployment-supplied VLM PromptPack (see
    # ``src.services.labeling.vlm_prompts.PromptPack.from_json`` and
    # ``docs/design/curation_design_rationale.md``'s PromptPack section).
    # ``None`` (the default) means "use the generic built-in pack" —
    # unlike the other paths on this dataclass there is no on-disk
    # default to fall back to, since most deployments never need one.
    prompt_pack_path: Path | None = None
    source_root: Path = Path('./data/images')
    export_root: Path = Path('./data/exports')
    source_path_aliases: Mapping[str, Path] = field(default_factory=dict)
    state_dir: Path = Path('/var/lib/openprocessor')
    crop_cache_dir: Path = Path('/dev/shm/openprocessor_crops')  # nosec B108 — intentional tmpfs cache

    api_prefix: str = '/curation'
    api_tag: str = 'Curation'

    embedding_dim: int = 512
    encoder_embedding_dim: int = 1024
    backbone_embedding_dim: int = 1024

    hnsw_ef_construction: int = 512
    hnsw_m: int = 16

    # Bake-off harness (src/routers/curation/bakeoff.py,
    # scripts/curation/bakeoff/). Previously hardcoded to owner-private
    # absolute paths (CFG-6) -- one of which named the location of a
    # licensed proprietary image corpus and must never appear in this repo
    # as a literal string. jobs/out mirror the state_dir/training_staging
    # precedent below; eval_root is a data root, so it mirrors
    # source_root/export_root instead.
    bakeoff_eval_root: Path = Path('./data/bakeoff_eval')

    @classmethod
    def from_env(cls, prefix: str = 'OP_') -> CurationConfig:
        """Build a :class:`CurationConfig` from ``{prefix}*`` env vars.

        Every field is optional; unset env vars fall back to the
        dataclass default. This is the OSS-facing override mechanism —
        deployment-specific overlays (like a future proprietary-dataset
        profile) are expected to construct the dataclass directly
        instead.
        """
        defaults = cls()

        def _str(name: str, default: str) -> str:
            return os.environ.get(f'{prefix}{name}', default)

        def _path(name: str, default: Path) -> Path:
            value = os.environ.get(f'{prefix}{name}')
            return Path(value) if value else default

        def _optional_path(name: str, default: Path | None) -> Path | None:
            value = os.environ.get(f'{prefix}{name}')
            return Path(value) if value else default

        def _int(name: str, default: int) -> int:
            value = os.environ.get(f'{prefix}{name}')
            return int(value) if value else default

        return cls(
            images_index=_str('IMAGES_INDEX', defaults.images_index),
            items_index=_str('ITEMS_INDEX', defaults.items_index),
            labels_confirmed_index=_str('LABELS_CONFIRMED_INDEX', defaults.labels_confirmed_index),
            classes_index=_str('CLASSES_INDEX', defaults.classes_index),
            clusters_index=_str('CLUSTERS_INDEX', defaults.clusters_index),
            settings_index=_str('SETTINGS_INDEX', defaults.settings_index),
            umap_state_index=_str('UMAP_STATE_INDEX', defaults.umap_state_index),
            umap_viz_state_index=_str('UMAP_VIZ_STATE_INDEX', defaults.umap_viz_state_index),
            class_registry_path=_path('REGISTRY_PATH', defaults.class_registry_path),
            prompt_pack_path=_optional_path('PROMPT_PACK_PATH', defaults.prompt_pack_path),
            source_root=_path('SOURCE_ROOT', defaults.source_root),
            export_root=_path('EXPORT_ROOT', defaults.export_root),
            source_path_aliases=defaults.source_path_aliases,
            state_dir=_path('STATE_DIR', defaults.state_dir),
            crop_cache_dir=_path('CROP_CACHE_DIR', defaults.crop_cache_dir),
            bakeoff_eval_root=_path('BAKEOFF_EVAL_ROOT', defaults.bakeoff_eval_root),
            api_prefix=_str('API_PREFIX', defaults.api_prefix),
            api_tag=_str('API_TAG', defaults.api_tag),
            embedding_dim=_int('EMBEDDING_DIM', defaults.embedding_dim),
            encoder_embedding_dim=_int('ENCODER_EMBEDDING_DIM', defaults.encoder_embedding_dim),
            backbone_embedding_dim=_int('BACKBONE_EMBEDDING_DIM', defaults.backbone_embedding_dim),
            hnsw_ef_construction=_int('HNSW_EF_CONSTRUCTION', defaults.hnsw_ef_construction),
            hnsw_m=_int('HNSW_M', defaults.hnsw_m),
        )


_INDEX_ROLE_ATTR: dict[IndexRole, str] = {
    IndexRole.IMAGES: 'images_index',
    IndexRole.ITEMS: 'items_index',
    IndexRole.LABELS_CONFIRMED: 'labels_confirmed_index',
    IndexRole.CLASSES: 'classes_index',
    IndexRole.SETTINGS: 'settings_index',
}


def index_name(cfg: CurationConfig, role: IndexRole) -> str:
    """Resolve the configured OpenSearch index name for a logical role.

    Replaces reference call sites of the shape ``KbIndex.X.value`` —
    those hardcoded a *value*; this looks the value up on ``cfg`` so it
    is deployment-overridable.
    """
    return getattr(cfg, _INDEX_ROLE_ATTR[role])


_default_curation_config: CurationConfig | None = None


def get_curation_config() -> CurationConfig:
    """Module-level default ``CurationConfig`` instance.

    Built via :meth:`CurationConfig.from_env` so the ``OP_*`` env vars
    documented on that classmethod (e.g. ``OP_API_PREFIX``) actually take
    effect for the process-wide default — this was previously
    constructing a bare ``CurationConfig()`` and silently ignoring every
    ``OP_*`` override.

    Callers that need a deployment-specific instance (e.g. a future
    overlay for an existing deployment) should construct and inject
    their own rather than relying on this default — mirrors
    :func:`src.config.region_fields.get_region_fields`.
    """
    global _default_curation_config  # noqa: PLW0603 - lazily-built module singleton
    if _default_curation_config is None:
        _default_curation_config = CurationConfig.from_env()
    return _default_curation_config

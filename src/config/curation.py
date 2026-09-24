"""Generic curation/labeling subsystem configuration.

Holds index names, filesystem roots and API-surface constants for the
`curation` namespace (OpenProcessor's generic port of a private
reference vehicle/license-plate curation stack — see
``docs/design/curation_design_rationale.md`` §2.1).

``CurationConfig`` replaces the module-level constants and the
``LegacyIndex`` string enum that the reference implementation hardcoded.
Index *values* are deployment data (a given operator's OpenSearch may
already have data under different index names), so they live on this
dataclass rather than in code. ``IndexRole`` + ``index_name()`` give a
stable, typo-proof way to look a configured index name up by its
logical role.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping


# Items-index field holding the per-item backbone embedding (a detector's
# feature map RoI-pooled over the item bbox, ``CurationConfig.
# backbone_embedding_dim`` wide). The mapping and the ingest writer both
# read this constant so they cannot drift; the name is kept for
# compatibility with existing indexes.
BACKBONE_EMBEDDING_FIELD = 'v6_embedding'


class IndexRole(str, Enum):
    """Logical role of a curation OpenSearch index.

    Unlike the reference ``LegacyIndex(str, Enum)``, member *values* are
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
    # Additional selectable packs (OP_PROMPT_PACK_PATHS, comma-separated),
    # advertised on GET /methods alongside the default pack and the
    # built-in generic pack; chosen per run / via the settings default.
    prompt_pack_paths: tuple[Path, ...] = ()
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
            prompt_pack_paths=tuple(
                Path(part.strip())
                for part in _str('PROMPT_PACK_PATHS', '').split(',')
                if part.strip()
            )
            or defaults.prompt_pack_paths,
            source_root=_path('SOURCE_ROOT', defaults.source_root),
            export_root=_path('EXPORT_ROOT', defaults.export_root),
            source_path_aliases=_parse_source_path_aliases(
                _str('SOURCE_PATH_ALIASES', ''), defaults.source_path_aliases
            ),
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


def _parse_source_path_aliases(raw: str, default: Mapping[str, Path]) -> Mapping[str, Path]:
    """Parse ``OP_SOURCE_PATH_ALIASES`` into ``{alias: root}``.

    Two accepted shapes: a JSON object (``{"archive": "/data/archive"}``)
    or a comma-separated ``alias=path`` list
    (``archive=/data/archive,nightly=/data/nightly``). Empty/unset keeps
    ``default``. Anything malformed raises ``ValueError`` — a silently
    dropped alias would 404 every image served through it.
    """
    raw = raw.strip()
    if not raw:
        return default
    pairs: dict[str, str]
    if raw.startswith('{'):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f'OP_SOURCE_PATH_ALIASES is not valid JSON: {exc}'
            raise ValueError(msg) from exc
        if not isinstance(loaded, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()
        ):
            msg = 'OP_SOURCE_PATH_ALIASES JSON must be an object of string alias -> string path'
            raise ValueError(msg)
        pairs = loaded
    else:
        pairs = {}
        for entry in (e.strip() for e in raw.split(',')):
            if not entry:
                continue
            alias, sep, path = entry.partition('=')
            if not sep:
                msg = f'OP_SOURCE_PATH_ALIASES entry {entry!r} must be alias=path'
                raise ValueError(msg)
            pairs[alias.strip()] = path.strip()
    aliases: dict[str, Path] = {}
    for alias, path in pairs.items():
        if not alias or not path or '/' in alias:
            msg = f'OP_SOURCE_PATH_ALIASES has an invalid alias/path pair: {alias!r}={path!r}'
            raise ValueError(msg)
        aliases[alias] = Path(path)
    return aliases


_INDEX_ROLE_ATTR: dict[IndexRole, str] = {
    IndexRole.IMAGES: 'images_index',
    IndexRole.ITEMS: 'items_index',
    IndexRole.LABELS_CONFIRMED: 'labels_confirmed_index',
    IndexRole.CLASSES: 'classes_index',
    IndexRole.SETTINGS: 'settings_index',
}


def index_name(cfg: CurationConfig, role: IndexRole) -> str:
    """Resolve the configured OpenSearch index name for a logical role.

    Replaces reference call sites of the shape ``LegacyIndex.X.value`` —
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

"""Generic curation/labeling subsystem configuration.

Holds index names, filesystem roots and API-surface constants for the
`curation` namespace (see
``docs/design/curation_design_rationale.md`` §2.1).

``CurationConfig`` replaces module-level constants and hardcoded
enum values with typed, deployment-configurable settings.
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
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Mapping


# Items-index field holding the per-item backbone embedding (a detector's
# feature map RoI-pooled over the item bbox, ``CurationConfig.
# backbone_embedding_dim`` wide). The mapping and the ingest writer both
# read this constant so they cannot drift.
BACKBONE_EMBEDDING_FIELD = 'backbone_embedding'

# Items-index field holding the per-item encoder embedding
# (``CurationConfig.encoder_embedding_dim`` wide). Queues that gate on "the
# item has been embedded" must use this name -- the items index has no
# generic ``embedding`` field.
ITEM_EMBEDDING_FIELD = 'pe_embedding'

# Minimum active-learning probe entropy (nats) for the 'all' review
# tab's catch-all clause. Before this, the tab matched on `exists
# probe_pred_entropy`, which after one probe run matches almost every
# non-holdout item -- a no-op filter in practice. Tune per-deployment;
# 1.0 nats is a reasonable default for a handful-of-classes cohort.
PROBE_ENTROPY_REVIEW_MIN = 1.0


class IndexRole(str, Enum):
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
    # The config store (W2, docs/design/openprocessor_internal/any_domain_plan.md
    # §3.1): prompt packs, region profiles, activations and the
    # cross-process revision counter. For a project created after W2,
    # ``resources_for_new`` also folds SETTINGS and UMAP_VIZ_STATE onto
    # this index's name (projects_plan.md §2.3, owner D4) -- ``default``
    # keeps its own separate settings/umap-viz-state indexes.
    CONFIGS = 'configs'


# Identity sentinel: "derive this path from ``state_dir``" (compared with
# ``is`` in ``CurationConfig.__post_init__``, never by value).
_FOLLOWS_STATE_DIR = Path('<follows state_dir>')
# "Take this from the ``default`` project's ``resources_for_new``" (an
# index name can never contain ``<``; the path is compared with ``is``).
_FROM_DEFAULT_PROJECT = '<from the default project>'
_FROM_DEFAULT_PROJECT_PATH = Path('<from the default project>')


@dataclass(frozen=True)
class CurationConfig:
    """Index names, filesystem roots and API-surface config for curation.

    Defaults are the generic OSS names. A deployment with existing data
    under different names (e.g. a proprietary-dataset overlay)
    constructs its own instance — see the design rationale in
    ``docs/design/curation_design_rationale.md`` §2.1. That overlay
    is not part of this generic module.
    """

    # Index names are never configured by env: every project's names are
    # ``{OP_PROJECT_INDEX_PREFIX}{slug}__{role}`` (``resources_for_new``),
    # resolved from the bound project. These fields exist so a caller can
    # construct an explicit ``CurationConfig`` (unit tests, one-off tools);
    # ``get_curation_config()`` always answers from the bound project. Left
    # unset, an explicit instance takes the ``default`` project's names and
    # data paths from ``resources_for_new`` (see __post_init__) -- there is
    # no second naming path.
    images_index: str = _FROM_DEFAULT_PROJECT
    items_index: str = _FROM_DEFAULT_PROJECT
    labels_confirmed_index: str = _FROM_DEFAULT_PROJECT
    classes_index: str = _FROM_DEFAULT_PROJECT
    # Single shared-defaults document (curation-strategy settings) — one
    # doc, not a full index of many rows. See
    # ``src.clients.curation_opensearch.CURATION_SETTINGS_DOC_ID`` for the
    # fixed doc id this index always addresses.
    # Shard folding (owner D4, projects_plan.md §2.3): SETTINGS folds onto
    # CONFIGS's name for every project, ``default`` included --
    # ``resources_for_new`` (via __post_init__ below) resolves the folded
    # name; there is no second naming path.
    settings_index: str = _FROM_DEFAULT_PROJECT
    # Two deliberately distinct UMAP-state indexes (see
    # ``src/services/curation/embedding_viz.py`` module docstring):
    # the retired clustering reducer's fitted-manifold cache
    # (``clustering/embedding_reduce.py``) and the visualization-only
    # projection's own metadata slot. UMAP_STATE keeps its own index;
    # UMAP_VIZ_STATE folds onto CONFIGS's name (shard folding, owner D4).
    umap_state_index: str = _FROM_DEFAULT_PROJECT
    umap_viz_state_index: str = _FROM_DEFAULT_PROJECT
    # Prompt packs, region profiles, activations, revision counter (W2).
    configs_index: str = _FROM_DEFAULT_PROJECT

    class_registry_path: Path = _FROM_DEFAULT_PROJECT_PATH
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
    export_root: Path = _FROM_DEFAULT_PROJECT_PATH
    # ST-4: keep-last retention for auto-named (timestamped) export dirs
    # under export_root. 0 = keep all. Never touches custom-named exports,
    # the `current` symlink target, or any dir pinned by a job/run/bake-off.
    export_keep_last: int = 5
    source_path_aliases: Mapping[str, Path] = field(default_factory=dict)
    state_dir: Path = Path('/var/lib/openprocessor')
    crop_cache_dir: Path = Path('/dev/shm/openprocessor_crops')  # nosec B108 — intentional tmpfs cache
    # ST-1: crop-cache prune threshold. write_crop_cache had no cap, so a
    # tmpfs mount without an OS-level size limit grows unbounded. 0 disables
    # pruning (source_image_cache.maybe_prune_crop_cache becomes a no-op).
    crop_cache_max_bytes: int = 3 * 1024**3
    # BA-1: server-managed, content-addressed root for uploaded image
    # bytes (POST /ingest/upload). Added to _configured_roots() (see
    # image_serving.py) so the path guards accept it. Default lives
    # under state_dir, following the crop_cache_dir precedent for a
    # server-owned data directory that isn't a mounted source archive.
    upload_root: Path = _FROM_DEFAULT_PROJECT_PATH

    # BA-2/BA-5: /ingest/upload + /ingest/batch request limits, served on
    # GET /ingest/config so a client never has to hardcode them.
    upload_max_images_per_request: int = 128
    upload_max_bytes_per_request: int = 512 * 1024 * 1024  # 512 MiB
    upload_accepted_extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    batch_max_items_per_request: int = 512

    api_prefix: str = '/curation'
    api_tag: str = 'Curation'

    embedding_dim: int = 512
    encoder_embedding_dim: int = 1024
    backbone_embedding_dim: int = 1024

    hnsw_ef_construction: int = 512
    hnsw_m: int = 16

    # Bake-off harness (src/routers/curation/bakeoff.py,
    # scripts/curation/bakeoff/). Previously hardcoded to owner-private
    # absolute paths -- one of which named the location of a
    # licensed proprietary image corpus and must never appear in this repo
    # as a literal string. jobs/out mirror the state_dir/training_staging
    # precedent below; eval_root is a data root, so it mirrors
    # source_root/export_root instead.
    bakeoff_eval_root: Path = _FROM_DEFAULT_PROJECT_PATH

    # Searchable per-item text (src/services/curation/item_text.py): the
    # region worker stores every OCR line read on the item crop. Effective
    # only when the active region profile names an OCR pipeline model.
    item_text_enabled: bool = True
    # Lines below this recognition score are not stored. 0.5 is PaddleOCR's
    # own ``drop_score`` default for its end-to-end system.
    item_text_min_confidence: float = 0.5

    # Browser-reachable MLflow base URL (e.g. ``https://mlflow.example.com``).
    # The trainer only ever sees ``MLFLOW_TRACKING_URI``, a container
    # hostname (e.g. ``http://curation-mlflow:5000``) unreachable from a
    # browser -- ``src/services/training/jobs.py`` rewrites the served
    # ``mlflow_run_url`` to use this base instead. ``None`` (the default)
    # means "no public MLflow UI configured" -- the served field is then
    # ``null`` rather than leaking the internal hostname.
    mlflow_public_url: str | None = None

    # Confidence floor gating the item wire's `probe_actionable` field
    # (see `src.services.curation.wire.serialize_item` and
    # `docs/design/curation_api_contract.md`). `probe_actionable` is true
    # only when the probe disagrees with the item's current class AND
    # the probe's top-1 posterior (`probe_pred_confidence`) is at least
    # this floor -- offering "accept model's class" when the probe itself
    # is barely more confident than a coin flip would be misleading.
    #
    # **Confidence, not entropy, is the gating signal.** `probe_pred_entropy`
    # (see `src.services.curation.probe_predictions`) is a raw Shannon
    # entropy in nats, bounded by `log(nc)` where `nc` is the probe
    # checkpoint's class count -- a value that varies across probe
    # versions/class-subsets and is not stored per item. A fixed threshold
    # against it would silently drift stricter or looser as `nc` changes.
    # `probe_pred_confidence` (the top-1 posterior after the sum-to-1
    # normalization in `probe_models._summarize_prediction_raw`) is always
    # in `[0, 1]` by construction regardless of `nc`, and is written in the
    # same bulk update as `probe_pred_class` (see `run_probe_inference`), so
    # it is reliably present whenever the probe has scored an item. 0.5
    # means the probe's top class holds a majority of the posterior mass --
    # a deployment-tunable bar, not a magic number.
    probe_actionable_min_confidence: float = 0.5

    # --- Project-scoped fields (see PROJECT_SCOPED_FIELDS below and
    # docs/design/openprocessor_internal/projects_plan.md §2.2/§3.3).
    # No env var sets any project-scoped field: ``get_curation_config()``
    # resolves them from the bound project's persisted resources. The
    # defaults only shape an explicitly constructed ``CurationConfig``
    # (unit tests, one-off tools). The two state paths follow
    # ``state_dir`` unless set explicitly (see __post_init__), so a config
    # built with ``state_dir=tmp`` never points at the real location.
    project_slug: str = 'default'
    project_state_dir: Path = _FOLLOWS_STATE_DIR
    train_jobs_dir: Path = Path('/jobs')
    autolabel_dir: Path = Path('/jobs/auto_label')
    bakeoff_jobs_dir: Path = _FOLLOWS_STATE_DIR
    mlflow_experiment: str = 'openprocessor'
    model_prefix: str = ''

    def __post_init__(self) -> None:
        unset_indexes = [
            attr
            for attr in _INDEX_ROLE_ATTR.values()
            if getattr(self, attr) == _FROM_DEFAULT_PROJECT
        ]
        unset_paths = [
            attr
            for attr in _DEFAULT_PROJECT_PATH_FIELDS
            if getattr(self, attr) is _FROM_DEFAULT_PROJECT_PATH
        ]
        if unset_indexes or unset_paths:
            from src.config.projects import DEFAULT_SLUG, resources_for_new

            resources = resources_for_new(DEFAULT_SLUG, self)
            role_of = {attr: role for role, attr in _INDEX_ROLE_ATTR.items()}
            for attr in unset_indexes:
                object.__setattr__(self, attr, resources.indexes[role_of[attr]])
            for attr in unset_paths:
                object.__setattr__(self, attr, getattr(resources, attr))
        if self.project_state_dir is _FOLLOWS_STATE_DIR:
            object.__setattr__(self, 'project_state_dir', self.state_dir)
        if self.bakeoff_jobs_dir is _FOLLOWS_STATE_DIR:
            object.__setattr__(self, 'bakeoff_jobs_dir', self.state_dir / 'bakeoff_jobs')

    @property
    def pause_sentinel_path(self) -> Path:
        """The ONE path every pause-sentinel writer/reader must agree on.

        The GPU arbiter (``src/services/training/gpu_arbiter.py``) used
        to default to ``{state_dir}/training_worker/pause.sentinel``
        while the workers that are supposed to pause when it appears
        (``scripts/curation/worker/state.py``,
        ``scripts/curation/vlm_worker.py``) defaulted to
        ``{state_dir}/vlm_worker/pause.sentinel`` -- a single-GPU
        training claim never actually paused anything, because the
        writer and the readers were watching two different files. All
        three now resolve through this property (env overrides on each
        side are unchanged and still work; they just need to agree if
        set).
        """
        return self.state_dir / 'vlm_worker' / 'pause.sentinel'

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

        def _optional_str(name: str, default: str | None) -> str | None:
            value = os.environ.get(f'{prefix}{name}')
            return value.strip() if value and value.strip() else default

        def _int(name: str, default: int) -> int:
            value = os.environ.get(f'{prefix}{name}')
            return int(value) if value else default

        def _float(name: str, default: float) -> float:
            value = os.environ.get(f'{prefix}{name}')
            return float(value) if value else default

        def _bool(name: str, default: bool) -> bool:
            value = os.environ.get(f'{prefix}{name}')
            if value is None or not value.strip():
                return default
            return value.strip().lower() in {'1', 'true', 'yes', 'on'}

        return cls(
            prompt_pack_path=_optional_path('PROMPT_PACK_PATH', defaults.prompt_pack_path),
            prompt_pack_paths=tuple(
                Path(part.strip())
                for part in _str('PROMPT_PACK_PATHS', '').split(',')
                if part.strip()
            )
            or defaults.prompt_pack_paths,
            source_root=_path('SOURCE_ROOT', defaults.source_root),
            export_keep_last=_int('EXPORT_KEEP_LAST', defaults.export_keep_last),
            source_path_aliases=_parse_source_path_aliases(
                _str('SOURCE_PATH_ALIASES', ''), defaults.source_path_aliases
            ),
            state_dir=_path('STATE_DIR', defaults.state_dir),
            crop_cache_dir=_path('CROP_CACHE_DIR', defaults.crop_cache_dir),
            crop_cache_max_bytes=_int('CROP_CACHE_MAX_BYTES', defaults.crop_cache_max_bytes),
            upload_max_images_per_request=_int(
                'UPLOAD_MAX_IMAGES_PER_REQUEST', defaults.upload_max_images_per_request
            ),
            upload_max_bytes_per_request=_int(
                'UPLOAD_MAX_BYTES_PER_REQUEST', defaults.upload_max_bytes_per_request
            ),
            upload_accepted_extensions=tuple(
                e.strip().lower()
                for e in _str(
                    'UPLOAD_ACCEPTED_EXTENSIONS', ','.join(defaults.upload_accepted_extensions)
                ).split(',')
                if e.strip()
            ),
            batch_max_items_per_request=_int(
                'BATCH_MAX_ITEMS_PER_REQUEST', defaults.batch_max_items_per_request
            ),
            api_prefix=_str('API_PREFIX', defaults.api_prefix),
            api_tag=_str('API_TAG', defaults.api_tag),
            mlflow_public_url=_optional_str('MLFLOW_PUBLIC_URL', defaults.mlflow_public_url),
            embedding_dim=_int('EMBEDDING_DIM', defaults.embedding_dim),
            encoder_embedding_dim=_int('ENCODER_EMBEDDING_DIM', defaults.encoder_embedding_dim),
            backbone_embedding_dim=_int('BACKBONE_EMBEDDING_DIM', defaults.backbone_embedding_dim),
            hnsw_ef_construction=_int('HNSW_EF_CONSTRUCTION', defaults.hnsw_ef_construction),
            hnsw_m=_int('HNSW_M', defaults.hnsw_m),
            item_text_enabled=_bool('ITEM_TEXT_ENABLED', defaults.item_text_enabled),
            item_text_min_confidence=_float(
                'ITEM_TEXT_MIN_CONFIDENCE', defaults.item_text_min_confidence
            ),
            probe_actionable_min_confidence=_float(
                'PROBE_ACTIONABLE_MIN_CONFIDENCE', defaults.probe_actionable_min_confidence
            ),
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


_DEFAULT_PROJECT_PATH_FIELDS = (
    'class_registry_path',
    'export_root',
    'upload_root',
    'bakeoff_eval_root',
)

_INDEX_ROLE_ATTR: dict[IndexRole, str] = {
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
    return getattr(cfg, _INDEX_ROLE_ATTR[role])


# Fields resolved from the *bound project*'s ``ProjectResources`` rather
# than the process-global env-built instance (see
# docs/design/openprocessor_internal/projects_plan.md §2.2/§3.3). Every
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
    zip(_INDEX_ROLE_ATTR.values(), _INDEX_ROLE_ATTR.keys(), strict=True)
)
# ^ inverts {IndexRole: attr_name} -> {attr_name: IndexRole}; both sides
# of ``_INDEX_ROLE_ATTR`` are unique so this round-trips exactly.

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

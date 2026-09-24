"""
Generic curation OpenSearch client.

Defines the four OpenSearch indexes used by the generic curation /
labeling subsystem (see ``docs/design/curation_design_rationale.md``
for the genericization rationale — this module is one of the
ratchet-exempt oversize files, §5), plus a ``ClassRegistry``
helper backed by an on-disk ``class_registry.json``.

Indexes (logical roles resolved via :func:`src.config.index_name`
against a :class:`~src.config.CurationConfig` instance — the actual
index *names* are deployment data, not hardcoded here):

- ``images`` (default ``op_images``) — one document per source image
  (with a global embedding).
- ``items`` (default ``op_items``) — one document per detected item
  crop (with embedding, class label, region-of-interest sub-bbox,
  holdout flag).
- ``labels_confirmed`` (default ``op_labels_confirmed``) — provenance
  ledger of imported YOLO-style ground-truth labels (no embedding),
  written only by label import. NOT the export source: export and
  training select ``class_validated=true`` items from ``items``, which
  every labeling path (human label/move, auto-promote, label import)
  sets — see ``tests/curation/test_labels_export_roundtrip.py``.
- ``classes`` (default ``op_classes``) — read-projection of
  ``class_registry.json`` for fast term filters / dashboards. The JSON
  file is the canonical source; this index is rebuilt from it via
  :py:meth:`ClassRegistry.sync_to_opensearch`.

Every OpenSearch field reference for the per-item "region of interest"
sub-annotation (e.g. a license plate on a vehicle crop) is routed
through the module-level :class:`~src.config.RegionFields` instance
(``F``) rather than hardcoded — see ``src/config/region_fields.py``
for the full design rationale. Everything else in the item schema
(``crop_id``, ``class_id``, clustering/scoring/probe fields, …) is
already deployment-agnostic and is not indirected.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from src.config import (
    BACKBONE_EMBEDDING_FIELD,
    CurationConfig,
    IndexRole,
    get_curation_config,
    get_region_fields,
    index_name,
)
from src.core.logging import get_logger
from src.services.curation.item_text import ITEM_TEXT_MAPPING


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# =============================================================================
# Module-level config singletons
# =============================================================================

config = get_curation_config()
F = get_region_fields()


# =============================================================================
# Pydantic models
# =============================================================================


class RegistryClassEntry(BaseModel):
    """Single class entry in ``class_registry.json``.

    The on-disk file uses ``id`` / ``name`` keys for backward
    compatibility with older label-file schemas. Internally we expose
    ``class_id`` / ``class_name`` to avoid shadowing Python builtins.
    Pydantic ``AliasChoices`` handles the bridge in both directions.

    Named ``RegistryClassEntry`` (not ``ClassEntry``) to avoid
    colliding with the distinct HTTP-response ``ClassEntry`` model in
    ``src.routers.curation._common``.
    """

    model_config = ConfigDict(populate_by_name=True)

    class_id: int = Field(validation_alias=AliasChoices('class_id', 'id'))
    class_name: str = Field(validation_alias=AliasChoices('class_name', 'name'))
    group: str = 'unknown'
    sample_count: int = 0
    validated_count: int = 0
    added_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        validation_alias=AliasChoices('added_at', 'added'),
    )
    deprecated: bool = False
    notes: str = ''
    merged_into: int | None = None  # populated when this class is merged into another
    # Optional single-character keyboard shortcut for fast assignment in the
    # labeler. Persisted in class_registry.json across sessions and devices.
    hotkey_letter: str | None = None


class ClassRegistryFile(BaseModel):
    """On-disk schema for ``class_registry.json``."""

    version: int = 1
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    classes: list[RegistryClassEntry] = Field(default_factory=list)


# =============================================================================
# Index schema builders
# =============================================================================


def _knn_field(dim: int = config.embedding_dim) -> dict[str, Any]:
    """Standard cosine k-NN HNSW field (FAISS engine)."""
    return {
        'type': 'knn_vector',
        'dimension': dim,
        'method': {
            'name': 'hnsw',
            'space_type': 'cosinesimil',
            'engine': 'faiss',
            'parameters': {
                'ef_construction': config.hnsw_ef_construction,
                'm': config.hnsw_m,
            },
        },
    }


def _knn_settings() -> dict[str, Any]:
    return {
        'index': {
            'number_of_shards': 1,
            'number_of_replicas': 0,
            'knn': True,
        },
    }


def _plain_settings() -> dict[str, Any]:
    return {
        'index': {
            'number_of_shards': 1,
            'number_of_replicas': 0,
        },
    }


def _images_body() -> dict[str, Any]:
    return {
        'settings': _knn_settings(),
        'mappings': {
            'properties': {
                'image_id': {'type': 'keyword'},
                'image_path': {'type': 'keyword'},
                'source': {'type': 'keyword'},
                'width': {'type': 'integer'},
                'height': {'type': 'integer'},
                'imohash': {'type': 'keyword'},
                'phash': {'type': 'keyword'},
                'indexed_at': {'type': 'date'},
                'original_resolution': {'type': 'keyword'},
                'non_jpeg_format_skipped': {'type': 'boolean'},
                'error_kind': {'type': 'keyword'},
                'embedding': _knn_field(),
                # Whole-frame secondary embedding, kNN-queryable for
                # whole-image near-duplicate detection and frame-level
                # similarity search. Distinct from the primary
                # `embedding` above. Requires index.knn=true (set by
                # _knn_settings); index.knn is a *final* setting in
                # OpenSearch, so enabling it on a pre-existing index
                # needs a reindex.
                'pe_embedding': _knn_field(dim=config.encoder_embedding_dim),
            }
        },
    }


# One entry per class write (src/services/curation/history.py). Human label
# writes record the full pre-write class state (class_detector* through
# cluster_subid, restorable=true) so the labeler's Undo restores it exactly.
#
# F-22: mapped as an unindexed object, not `nested`. Nothing ever issues a
# `nested` query or agg against this field (only mapping + plain `_source`
# reads/writes) -- rg -n "'nested'" src scripts turns up none -- yet every
# entry cost a hidden Lucene doc (the reference index carried 560k Lucene
# docs for 348k items), every write rewrote the whole nested block, and every
# top-level query without a positive clause picked up a `FieldExistsQuery
# [_primary_term]` parent filter (measured 21-105ms of query time). `enabled:
# False` keeps the data in `_source` (label_undo.py reads `_source`, not the
# mapping) without indexing any of it.
_CLASS_HISTORY_MAPPING: dict[str, Any] = {
    'type': 'object',
    'enabled': False,
}

# Exclusion plus the other per-item human review decisions.
_EXCLUSION_MAPPING: dict[str, Any] = {
    'class_excluded': {'type': 'boolean'},
    'excluded_at': {'type': 'date'},
    'excluded_by': {'type': 'keyword'},
    'excluded_reason': {'type': 'keyword'},
    'excluded_prior_class_validated': {'type': 'boolean'},
    'excluded_prior_cluster_id': {'type': 'integer'},
    'excluded_prior_cluster_subid': {'type': 'keyword'},
    # Review-queue dismissal (POST /crops/{id}/review_dismiss, /discard).
    'review_dismissed_at': {'type': 'date'},
    'review_dismissed_by': {'type': 'keyword'},
    # Rejected VLM suggestion (POST /crops/{id}/vlm_dismiss).
    'vlm_dismissed_class_id': {'type': 'integer'},
    'vlm_dismissed_class_name': {'type': 'keyword'},
    'vlm_dismissed_at': {'type': 'date'},
    # Undo snapshots of region writes + VLM dismissals
    # (src/services/curation/edit_history.py). Stored, never indexed:
    # entries hold arbitrary-typed field snapshots and are only ever read
    # back whole by the undo routes.
    'edit_history': {'type': 'object', 'enabled': False},
}


def _region_text_reader_mapping() -> dict[str, Any]:
    """Per-reader region text fields (VLM vs OCR reading + disagreement)."""
    return {
        F.text_vlm: {'type': 'keyword'},
        F.text_ocr: {'type': 'keyword'},
        F.text_disagreement: {'type': 'boolean'},
    }


def _items_body() -> dict[str, Any]:
    return {
        'settings': _knn_settings(),
        'mappings': {
            'properties': {
                'crop_id': {'type': 'keyword'},
                'image_id': {'type': 'keyword'},
                'image_path': {'type': 'keyword'},
                'source': {'type': 'keyword'},
                # X-Request-ID propagated from the HTTP ingest call (or
                # '-' for non-HTTP callers). Lets operators correlate
                # worker / labeler logs to the originating ingest
                # request when triaging stuck items.
                'request_id': {'type': 'keyword'},
                'bbox_norm': {'type': 'float'},  # 4-element array [x1, y1, x2, y2] (normalized)
                'class_id': {'type': 'integer'},
                'class_name': {'type': 'keyword'},
                'class_source': {'type': 'keyword'},
                # VLM's raw answer for every classification call (whether or
                # not it resolved against the registry). Aggregating this field
                # via terms agg surfaces the long-tail labels that should grow
                # the registry to cover. Optional confidence float (0-1) is
                # written when the VLM supplies a numeric score; otherwise the
                # bucketed confidence keyword carries the signal.
                #
                # Field name kept as-is (not indirected via RegionFields —
                # out of scope, §3.2 scope table): this is a live persisted
                # OpenSearch key, and only the region-of-interest ("plate")
                # fields have an indirection mechanism in Phase 2.
                'vlm_raw_label': {'type': 'keyword'},
                'vlm_raw_label_conf': {'type': 'float'},
                # Marker: class + region resolved in one combined VLM call
                # (scripts/curation/worker/verify.py). Downstream pipeline
                # stages range-query this to skip a redundant class call
                # (src/services/curation/autolabel/selection.py).
                'vlm_verify_completed_at': {'type': 'date'},
                # VLM-extracted make/model hint. Field names kept as-is for
                # the same reason as above (no region-of-interest concept
                # applies to a vehicle make/model).
                'vlm_item_make': {'type': 'keyword'},
                'vlm_item_model': {'type': 'keyword'},
                # Region-visibility hint (CFG-8): this WAS a domain-named,
                # vendor-named field ('gemma_plate_visible') baked into the
                # otherwise-generic index mapping, unlike its siblings above
                # it IS a region-of-interest concept and RegionFields
                # already has an indirection for it -- see
                # RegionFields.visible (default 'region_visible').
                F.visible: {'type': 'boolean'},
                # Hierarchical clustering of vlm_raw_label values. A
                # background job writes back a cluster id (stable hash of the
                # cluster name) and the human-readable cluster name so a
                # review UI can group fine-grained sub-classes and suggest
                # registry promotions.
                'vlm_label_cluster_id': {'type': 'integer'},
                'vlm_label_cluster_name': {'type': 'keyword'},
                'vlm_label_cluster_distance': {'type': 'float'},
                'confidence': {'type': 'float'},
                'cluster_id': {'type': 'integer'},
                'cluster_distance': {'type': 'float'},
                'cluster_subid': {'type': 'keyword'},  # AHC sub-cluster id (e.g. "47a")
                # Region-of-interest clustering — independent of the item
                # cluster_* above (an item's region rides on an item that
                # already owns those). Coarse IVF partition + per-bucket AHC
                # refine over the region embedding, so region
                # false-positives / bad boxes surface as sub-cluster
                # outliers.
                F.cluster_id: {'type': 'integer'},
                F.cluster_distance: {'type': 'float'},
                F.cluster_subid: {'type': 'keyword'},
                'cluster_auto_suggest': {'type': 'keyword'},
                # Primary-subject ranking + blur quality (computed at ingest,
                # backfilled for legacy items). crop_rank_in_image=1 is the
                # largest-area item in its source photo. blur_lap_ratio is the
                # Laplacian crop/full ratio; blur_lap_var the raw crop
                # Laplacian variance. Used by the primary-subject label filters
                # and the optional clustering gate.
                'crop_area_norm': {'type': 'float'},
                'crop_rank_in_image': {'type': 'byte'},
                'blur_lap_var': {'type': 'float'},
                'blur_lap_ratio': {'type': 'float'},
                'blur_full_var': {'type': 'float'},
                # `label_validated` is the LEGACY conflated flag. New
                # writers should use class_validated / the region
                # ``validated`` field instead. The legacy field is kept
                # readable for one release as a derived passthrough for
                # transitional frontend consumers.
                'label_validated': {'type': 'boolean'},
                'class_validated': {'type': 'boolean'},
                F.validated: {'type': 'boolean'},
                'label_source': {'type': 'keyword'},
                F.bbox_norm: {'type': 'float'},
                F.score: {'type': 'float'},
                F.verified: {'type': 'boolean'},
                F.reason: {'type': 'text'},
                # Region-of-interest provenance. Every region write carries
                # which detector produced the bbox, the detector version, the
                # bbox coordinate frame, and when it was written. Verifier
                # fields are populated when a VLM (or a human) confirmed the
                # candidate. ``detector_chain`` is a multi-value keyword
                # (OpenSearch arrays of keyword work as-is).
                F.detector: {'type': 'keyword'},
                F.detector_version: {'type': 'keyword'},
                F.detector_chain: {'type': 'keyword'},
                F.bbox_frame: {'type': 'keyword'},
                F.detected_at: {'type': 'date'},
                F.verifier: {'type': 'keyword'},
                F.verifier_version: {'type': 'keyword'},
                F.verified_at: {'type': 'date'},
                F.rejection_reason: {'type': 'keyword'},
                # Region lifecycle + VLM read-back. Explicit so none of these
                # fall to dynamic `text` mapping, where terms aggregations and
                # sorts on the bare field name fail.
                F.status: {'type': 'keyword'},
                F.bbox_correct: {'type': 'boolean'},
                F.confidence: {'type': 'keyword'},
                F.text: {'type': 'keyword', 'fields': {'search': {'type': 'text'}}},
                F.text_raw: {'type': 'keyword'},
                F.text_confidence: {'type': 'float'},
                F.text_source: {'type': 'keyword'},
                F.text_engine_version: {'type': 'keyword'},
                **_region_text_reader_mapping(),
                # Every OCR line read on the item crop + normalized search
                # tokens (src/services/curation/item_text.py).
                **ITEM_TEXT_MAPPING,
                F.class_id: {'type': 'integer'},
                F.label_source: {'type': 'keyword'},
                F.source: {'type': 'keyword'},
                F.pairing: {'type': 'keyword'},
                F.skip_verify: {'type': 'boolean'},
                # Item label fields written by ingest and the VLM labeler.
                'proposal_name': {'type': 'keyword'},
                'vlm_confidence': {'type': 'keyword'},
                'vlm_raw_class': {'type': 'keyword'},
                'vlm_proposed_class': {'type': 'keyword'},
                'needs_new_class': {'type': 'boolean'},
                # Quarantine bookkeeping — a legacy-region quarantine script
                # copies the original values into these fields before
                # clearing the live fields and re-running detection.
                F.bbox_norm_legacy: {'type': 'float'},
                F.score_legacy: {'type': 'float'},
                F.status_legacy: {'type': 'keyword'},
                # Class-label provenance (same shape as region provenance,
                # scoped to the item class label rather than the
                # region-of-interest sub-bbox).
                'class_detector': {'type': 'keyword'},
                'class_detector_version': {'type': 'keyword'},
                'class_labeler': {'type': 'keyword'},
                'class_labeled_at': {'type': 'date'},
                'test_holdout': {'type': 'boolean'},
                'probe_pred_class': {'type': 'keyword'},
                # Registry id of probe_pred_class (null if not in the registry).
                'probe_pred_class_id': {'type': 'integer'},
                'probe_pred_entropy': {'type': 'float'},
                # Probe-model provenance + real per-class posterior
                # derivatives.
                'probe_pred_confidence': {'type': 'float'},
                'probe_disagreement': {'type': 'boolean'},
                'probe_pred_margin': {'type': 'float'},
                'probe_model_version': {'type': 'keyword'},
                'probe_scored_at': {'type': 'date'},
                # Curation-scoring overlay fields. Written ONLY by the
                # dedicated scoring job (src/services/curation/item_scores/);
                # never by the production clustering pipeline. No overlay
                # ever writes cluster_id / cluster_subid / cluster_distance.
                'uniqueness_score': {'type': 'float'},
                'uniqueness_method': {'type': 'keyword'},
                'uniqueness_version': {'type': 'keyword'},
                'uniqueness_scored_at': {'type': 'date'},
                'mistakenness_score': {'type': 'float'},
                'mistakenness_method': {'type': 'keyword'},
                'mistakenness_version': {'type': 'keyword'},
                'mistakenness_scored_at': {'type': 'date'},
                'dup_group_id': {'type': 'keyword'},
                'dup_group_size': {'type': 'integer'},
                'dup_is_representative': {'type': 'boolean'},
                'dup_threshold': {'type': 'float'},
                'dup_method': {'type': 'keyword'},
                'dup_scored_at': {'type': 'date'},
                'created_at': {'type': 'date'},
                'updated_at': {'type': 'date'},
                # Secondary image-encoder embedding for semantic search
                # (e.g. "white RV", "ford f150").
                'pe_embedding': _knn_field(dim=config.encoder_embedding_dim),
                # Backbone RoI-pool embedding used for residual AHC
                # clustering + intra-class similarity refinement.
                BACKBONE_EMBEDDING_FIELD: _knn_field(dim=config.backbone_embedding_dim),
                # Encoder embedding of the region-of-interest (cropped at
                # the region bbox, pad-to-square). Lets regions be
                # clustered / AHC-refined like item classes so
                # false-positives and bad boxes surface as outliers.
                # F-23/D-2: knn_vector, not a plain indexed float array. A
                # 1024-value float array indexed 1024 BKD points plus
                # useless sorted/deduplicated doc values and stored ~22KiB
                # of JSON in _source per doc (measured fetch cost 50-70ms
                # per 60 docs even with it _source-excluded, since derived
                # source still has to skip past it). knn_vector gets
                # binary derived source instead of JSON and becomes
                # kNN-searchable for region FP matching. The previous
                # comment here claimed index.knn was disabled on live
                # indexes -- that's no longer true (op_items has
                # index.knn: true), so the plain-float rationale no
                # longer applies.
                F.embedding: _knn_field(dim=config.encoder_embedding_dim),
                # History: nested array recording every class write so
                # operators can answer "who labeled this and when" after a
                # model drift investigation. Cap at MAX_HISTORY_ENTRIES (32,
                # see src/services/curation/history.py).
                'class_id_history': _CLASS_HISTORY_MAPPING,
                # Label Ignore/Undo: exclusion flag + provenance, and the
                # pre-exclusion validation/cluster placement un-exclude
                # restores (src/services/curation/exclusion.py).
                **_EXCLUSION_MAPPING,
            }
        },
    }


def _labels_confirmed_body() -> dict[str, Any]:
    return {
        'settings': _plain_settings(),
        'mappings': {
            'properties': {
                'label_id': {'type': 'keyword'},
                'image_path': {'type': 'keyword'},
                'bbox_norm': {'type': 'float'},
                'class_id': {'type': 'integer'},
                'class_name': {'type': 'keyword'},
                'label_source': {'type': 'keyword'},
                'confirmed_at': {'type': 'date'},
                'crop_id': {'type': 'keyword'},
            }
        },
    }


def _classes_body() -> dict[str, Any]:
    return {
        'settings': _plain_settings(),
        'mappings': {
            'properties': {
                'class_id': {'type': 'integer'},
                'class_name': {'type': 'keyword'},
                'group': {'type': 'keyword'},
                'sample_count': {'type': 'long'},
                'validated_count': {'type': 'long'},
                'added_at': {'type': 'date'},
                'deprecated': {'type': 'boolean'},
                'notes': {'type': 'text'},
            }
        },
    }


def _settings_body() -> dict[str, Any]:
    """Curation-strategy shared-defaults document (one row, doc id
    :data:`CURATION_SETTINGS_DOC_ID`) -- backs ``GET/PUT /curation/settings``
    and :func:`~src.services.curation.strategy_registry.resolve_effective_default`.

    ``defaults`` is deliberately ``enabled: false`` (stored, never
    indexed/searchable) rather than a strict per-axis mapping: it is an
    OPEN map keyed by axis id (a future axis must not require a mapping
    change / reindex), and nothing ever queries into it -- every read is
    a single ``GET`` by the fixed doc id, never a search. OpenSearch's
    partial ``update`` API still does its normal recursive object merge
    against `_source` regardless of ``enabled``, which is exactly what a
    partial ``PUT /curation/settings`` needs (merge one axis in without
    clobbering the others).
    """
    return {
        'settings': _plain_settings(),
        'mappings': {
            'properties': {
                'defaults': {'type': 'object', 'enabled': False},
                'updated_at': {'type': 'date'},
                'updated_by': {'type': 'keyword'},
            }
        },
    }


def _umap_state_body() -> dict[str, Any]:
    """The retired clustering reducer's fitted-manifold cache
    (``clustering/embedding_reduce.py``). ``reducer_b64`` is a pickled
    UMAP reducer, base64-encoded, up to ~60 MB (see
    ``_OPENSEARCH_PERSIST_MAX_BYTES``) -- mapped ``binary`` (stored,
    never analyzed/indexed) rather than left to dynamic mapping, which
    tokenized it as ``text`` (F-27)."""
    return {
        'settings': _plain_settings(),
        'mappings': {
            'dynamic': False,
            'properties': {
                'state_id': {'type': 'keyword'},
                'reducer_b64': {'type': 'binary'},
                'n_components': {'type': 'integer'},
                'metric': {'type': 'keyword'},
            },
        },
    }


def _umap_viz_state_body() -> dict[str, Any]:
    """Visualization-only projection's own metadata slot
    (``src/services/curation/embedding_viz.py``) -- deliberately
    distinct from :func:`_umap_state_body`. Metadata only, no pickled
    blob."""
    return {
        'settings': _plain_settings(),
        'mappings': {
            'dynamic': False,
            'properties': {
                'state_id': {'type': 'keyword'},
                'projection_version': {'type': 'keyword'},
                'scope': {'type': 'keyword'},
                'cluster_id': {'type': 'integer'},
                'n_points': {'type': 'integer'},
                'fitted_at': {'type': 'date'},
                'n_components': {'type': 'integer'},
                'metric': {'type': 'keyword'},
            },
        },
    }


INDEX_BODIES: dict[IndexRole, dict[str, Any]] = {
    IndexRole.IMAGES: _images_body(),
    IndexRole.ITEMS: _items_body(),
    IndexRole.LABELS_CONFIRMED: _labels_confirmed_body(),
    IndexRole.CLASSES: _classes_body(),
    IndexRole.SETTINGS: _settings_body(),
    IndexRole.UMAP_STATE: _umap_state_body(),
    IndexRole.UMAP_VIZ_STATE: _umap_viz_state_body(),
}


CURATION_SETTINGS_DOC_ID = 'default'
"""Fixed OpenSearch doc id the settings index always addresses -- this is a
single shared-defaults document, not a full index of many settings rows
(curation_design_rationale.md's config-dataclass philosophy: one small,
explicit piece of deployment/runtime state, not a generic key-value
store). ``'default'`` (not e.g. ``'singleton'``) because it reads naturally
alongside the field it stores (\"the defaults doc\"), and because a future
per-tenant settings doc (if this ever stops being a single shared
instance) would key by tenant id with this same literal as the
single-tenant fallback."""


# F-28.1: get_curation_settings is read on nearly every strategy-scoring
# request path (strategy_defaults.py, strategy_registry.py both fetch it
# per call). A 5s TTL cache avoids a GET-by-id round trip on every one of
# those, while staying short enough that a settings change is visible
# almost immediately -- and update_curation_settings below invalidates it
# immediately on write anyway, so the TTL only matters between writes.
# Keyed by index name so a caller passing a non-default cfg doesn't share
# another deployment's cached doc.
_SETTINGS_CACHE_TTL_SECONDS = 5.0
_settings_cache: dict[str, tuple[dict[str, Any], float]] = {}


def _invalidate_settings_cache(index: str) -> None:
    _settings_cache.pop(index, None)


async def get_curation_settings(client: Any, cfg: CurationConfig | None = None) -> dict[str, Any]:
    """Fetch the shared curation-settings document.

    Get-or-default-empty: a missing document (nothing has ever been PUT)
    is not an error -- it means "no shared override for any axis yet" --
    so this always returns the full envelope shape with ``defaults: {}``
    rather than raising or returning ``None``.

    An axis explicitly cleared via ``update_curation_settings(..., {axis:
    None})`` is stored as a literal ``null`` (OpenSearch's partial-doc
    merge sets a nested field to null rather than deleting the key) --
    filtered out here so a cleared axis simply doesn't appear in
    ``defaults``, identical to "never had an override."

    F-28.1: cached for :data:`_SETTINGS_CACHE_TTL_SECONDS`, invalidated
    immediately by :func:`update_curation_settings` on write.
    """
    active_cfg = cfg or config
    index = index_name(active_cfg, IndexRole.SETTINGS)

    cached = _settings_cache.get(index)
    if cached is not None and time.monotonic() < cached[1]:
        return cached[0]

    source: dict[str, Any] = {}
    try:
        resp = await client.get(index=index, id=CURATION_SETTINGS_DOC_ID)
        source = resp.get('_source') or {} if isinstance(resp, dict) else {}
    except Exception as exc:
        # Mirrors image_serving.fetch_crop_source's duck-typed not-found
        # check -- avoids a hard opensearchpy import just to catch
        # NotFoundError, so a plain test mock with a raising `.get` works
        # the same way the real client does.
        msg = str(exc).lower()
        if not ('notfound' in msg or 'not found' in msg or '404' in msg):
            logger.warning('curation_settings_get_failed', error=str(exc))
    raw_defaults = source.get('defaults') or {}
    result = {
        'defaults': {k: v for k, v in raw_defaults.items() if v is not None},
        'updated_at': source.get('updated_at'),
        'updated_by': source.get('updated_by'),
    }
    _settings_cache[index] = (result, time.monotonic() + _SETTINGS_CACHE_TTL_SECONDS)
    return result


async def update_curation_settings(
    client: Any, defaults: dict[str, str | None], cfg: CurationConfig | None = None
) -> dict[str, Any]:
    """Partially merge ``defaults`` into the single shared settings doc.

    Uses OpenSearch's partial-update ``doc`` merge (recursive for object
    fields, per the update API's documented semantics) so axes not
    mentioned in this call are left untouched -- callers never need to
    read-modify-write the whole document themselves. ``doc_as_upsert``
    creates the document on the very first write. ``updated_by`` stays
    ``None`` -- there is no user-account system yet (single shared
    instance) -- but the field is written on every call so the schema
    already carries it for when one exists.

    A ``None`` value for an axis clears its shared override -- stored as
    a literal null (see :func:`get_curation_settings`'s note on why that
    read path filters it back out).

    F-28.1: no ``refresh=True`` -- the read-immediately-after-write below
    is a single-doc ``GET`` (not ``_search``), which OpenSearch serves
    real-time from the translog regardless of the index's refresh
    interval, so forcing a segment refresh here bought nothing but
    latency. The 5s settings cache is invalidated immediately (not left
    to expire) so this read-after-write can never return a stale value.
    """
    active_cfg = cfg or config
    index = index_name(active_cfg, IndexRole.SETTINGS)
    body = {
        'doc': {
            'defaults': defaults,
            'updated_at': datetime.now(UTC).isoformat(),
            'updated_by': None,
        },
        'doc_as_upsert': True,
    }
    await client.update(index=index, id=CURATION_SETTINGS_DOC_ID, body=body)
    _invalidate_settings_cache(index)
    return await get_curation_settings(client, cfg=active_cfg)


async def get_curation_index_settings() -> dict[str, dict[str, Any]]:
    """Return the schema dict keyed by index name (string).

    Used by tests + introspection routes.
    """
    return {index_name(config, role): body for role, body in INDEX_BODIES.items()}


# =============================================================================
# Index lifecycle
# =============================================================================


async def _create_one(
    client: AsyncOpenSearch,
    index: str,
    body: dict[str, Any],
    force_recreate: bool,
) -> bool:
    """Create a single index idempotently."""
    try:
        exists = await client.indices.exists(index=index)
        if exists:
            if force_recreate:
                logger.info('curation_index_delete', index=index)
                await client.indices.delete(index=index)
            else:
                logger.info('curation_index_exists', index=index)
                return True
        try:
            await client.indices.create(index=index, body=body)
            logger.info('curation_index_created', index=index)
        except Exception as create_err:
            # Concurrent ingest workers can race here: exists() returns False
            # for everyone, then they all try to create. OpenSearch returns
            # resource_already_exists_exception for the losers -- treat it
            # as success to keep logs clean and behavior idempotent.
            if 'resource_already_exists_exception' in str(create_err):
                logger.debug('curation_index_create_race_won_by_peer', index=index)
            else:
                raise
        return True
    except Exception as e:
        logger.error('curation_index_create_failed', index=index, error=str(e))
        return False


def _is_recoverable_mapping_conflict(msg: str) -> bool:
    """Classify a ``put_mapping`` failure as permanent-and-harmless vs. real.

    The ``ensure_items_*`` helpers below run on every cold start to
    additively backfill fields onto long-lived deployments. Two error
    shapes are BOTH expected, permanent conditions once a live index has
    drifted from the current schema code — not a fresh incident each
    restart, and never fatal to boot:

    * ``mapper_parsing_exception`` / "already exists" — a genuine field
      name collision (should not happen in production, but is harmless).
    * ``illegal_argument_exception`` — OpenSearch's error type for two
      immutable-setting conflicts: a field whose type already differs from
      what an ``ensure_*`` helper wants (a type change requires a full
      reindex), and setting kNN method params on an index whose
      ``index.knn`` is still ``false`` (pre-kNN-migration indices). Both
      require a reindex to actually resolve, not a retry — logging them at
      ``error`` on every single restart would be misleading operational
      noise, not an indication anything was newly broken.
    """
    return (
        'mapper_parsing_exception' in msg
        or 'already exists' in msg
        or 'illegal_argument_exception' in msg
    )


async def ensure_items_vlm_raw_label_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the ``vlm_raw_label`` + ``vlm_raw_label_conf`` fields onto the
    existing items mapping.

    OpenSearch ``PUT <index>/_mapping`` is idempotent for additive field
    changes — running it repeatedly is a no-op once the fields exist. This
    helper exists so deployments don't have to drop / recreate the index just
    to start capturing the VLM's raw output.

    Returns:
        Dict with ``acknowledged`` (bool from OpenSearch) plus ``index`` and
        ``fields_added`` (the field names this call attempted to add).
    """
    index = config.items_index
    body = {
        'properties': {
            'vlm_raw_label': {'type': 'keyword'},
            'vlm_raw_label_conf': {'type': 'float'},
            'vlm_verify_completed_at': {'type': 'date'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties'].keys()),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties'].keys()),
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties'].keys()),
            'error': msg,
        }


async def ensure_items_label_cluster_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the ``vlm_label_cluster_*`` fields onto the existing items mapping.

    Why a dedicated helper: ``PUT <index>/_mapping`` is idempotent for
    additive field changes, so this can run on every cold start without
    triggering an index drop. A background clustering job calls this before
    writing back cluster ids to guarantee the fields exist on long-lived
    deployments created before this migration shipped.

    Returns:
        Dict with ``acknowledged`` / ``index`` / ``fields_added`` (and
        ``error`` on failure).
    """
    index = config.items_index
    body = {
        'properties': {
            'vlm_label_cluster_id': {'type': 'integer'},
            'vlm_label_cluster_name': {'type': 'keyword'},
            'vlm_label_cluster_distance': {'type': 'float'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties'].keys()),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties'].keys()),
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties'].keys()),
            'error': msg,
        }


async def ensure_items_provenance_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the region + class provenance fields onto the existing items
    mapping.

    Additive ``PUT <index>/_mapping`` — idempotent: re-running on a mapping
    that already has the fields is a silent no-op. The list of fields here
    mirrors the entries added to :func:`_items_body`, so a freshly created
    index already has them and this call simply confirms. Adds for
    long-lived deployments that were created before this migration shipped.

    ``mapper_parsing_exception`` (raised on the rare case where a field
    name collides with an incompatible existing type — should never happen
    in production but is the documented failure mode) is swallowed with a
    log line; this helper never crashes the cold-start bootstrap.
    """
    index = config.items_index
    body = {
        'properties': {
            F.detector: {'type': 'keyword'},
            F.detector_version: {'type': 'keyword'},
            F.detector_chain: {'type': 'keyword'},
            F.bbox_frame: {'type': 'keyword'},
            F.detected_at: {'type': 'date'},
            F.verifier: {'type': 'keyword'},
            F.verifier_version: {'type': 'keyword'},
            F.verified_at: {'type': 'date'},
            F.rejection_reason: {'type': 'keyword'},
            F.bbox_norm_legacy: {'type': 'float'},
            F.score_legacy: {'type': 'float'},
            F.status_legacy: {'type': 'keyword'},
            'class_detector': {'type': 'keyword'},
            'class_detector_version': {'type': 'keyword'},
            'class_labeler': {'type': 'keyword'},
            'class_labeled_at': {'type': 'date'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties'].keys()),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties'].keys()),
        }
    except Exception as exc:
        # OpenSearch's ``mapper_parsing_exception`` arrives wrapped in an
        # ``opensearchpy`` transport error; we don't import the client class
        # here so we string-match the error type. Either way, the failure
        # is logged and the bootstrap continues — additive mapping
        # collisions are non-fatal for the rest of the API surface.
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties'].keys()),
            'error': msg,
        }


async def ensure_items_validation_split_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the split ``class_validated`` + region ``validated`` boolean
    fields onto the existing items mapping.

    The conflated ``label_validated`` field is being split because a
    worker's region writes were silently un-validating human class labels
    by sharing the flag. The new fields are independent: a region-edit
    write touches only the region ``validated`` field, a class-edit write
    touches only ``class_validated``.

    Additive PUT — idempotent. The legacy ``label_validated`` field
    is preserved during the one-release deprecation window; new
    writers SHOULD NOT set it (the pre-commit guard blocks new writes).
    """
    index = config.items_index
    fields = ['class_validated', F.validated]
    body = {
        'properties': {
            'class_validated': {'type': 'boolean'},
            F.validated: {'type': 'boolean'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': fields,
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': fields,
            'error': msg,
        }


async def ensure_items_history_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the ``class_id_history`` object field onto the existing items
    mapping.

    Additive ``PUT <index>/_mapping`` — idempotent. Writers land in
    ``src/services/curation/history.py``; this helper exists so a
    re-ingested or migrated index has the field ready when the writers go
    live.

    F-22: a field's type can't change in place — an index built before this
    field went from ``nested`` to ``object enabled:false`` still has it
    mapped ``nested``, and OpenSearch would 400 on a conflicting
    ``put_mapping`` every cold start. No-op whenever the field is already
    present, regardless of its type; the type change itself only takes
    effect on a reindex (see the F-5 migration note).
    """
    index = config.items_index
    try:
        existing = await client.indices.get_mapping(index=index)
    except Exception as exc:
        logger.info('curation_mapping_precheck_failed', index=index, error=str(exc))
        existing = {}
    for mapping in (existing or {}).values():
        if 'class_id_history' in (mapping.get('mappings', {}).get('properties') or {}):
            return {
                'acknowledged': True,
                'index': index,
                'fields_added': [],
                'skipped': 'field_already_present',
            }
    body = {
        'properties': {
            'class_id_history': _CLASS_HISTORY_MAPPING,
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties']),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties']),
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties']),
            'error': msg,
        }


async def ensure_items_exclusion_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the exclusion fields (:data:`_EXCLUSION_MAPPING`) onto the items mapping.

    One ``PUT _mapping`` per field: indexes created before these were
    mapped may already carry a dynamic ``text`` mapping for the string
    fields, and a conflict on one must not block the others.
    """
    index = config.items_index
    added: list[str] = []
    conflicts: list[str] = []
    for field, spec in _EXCLUSION_MAPPING.items():
        try:
            await client.indices.put_mapping(index=index, body={'properties': {field: spec}})
            added.append(field)
        except Exception as exc:
            msg = str(exc)
            if not _is_recoverable_mapping_conflict(msg):
                logger.error('curation_mapping_migration_failed', index=index, error=msg)
                return {'acknowledged': False, 'index': index, 'fields_added': added, 'error': msg}
            conflicts.append(field)
    logger.info(
        'curation_mapping_migration', index=index, fields=added, existing_conflicts=conflicts
    )
    return {'acknowledged': True, 'index': index, 'fields_added': added, 'conflicts': conflicts}


async def ensure_items_text_reader_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the region text-reader fields and the item-text fields onto the
    items mapping.

    One ``PUT _mapping`` per field (as :func:`ensure_items_exclusion_fields`)
    so a dynamic mapping one of them already picked up on an older index
    cannot block the others. Additive and idempotent.
    """
    index = config.items_index
    added: list[str] = []
    conflicts: list[str] = []
    for field, spec in {**_region_text_reader_mapping(), **ITEM_TEXT_MAPPING}.items():
        try:
            await client.indices.put_mapping(index=index, body={'properties': {field: spec}})
            added.append(field)
        except Exception as exc:
            msg = str(exc)
            if not _is_recoverable_mapping_conflict(msg):
                logger.error('curation_mapping_migration_failed', index=index, error=msg)
                return {'acknowledged': False, 'index': index, 'fields_added': added, 'error': msg}
            conflicts.append(field)
    logger.info(
        'curation_mapping_migration', index=index, fields=added, existing_conflicts=conflicts
    )
    return {'acknowledged': True, 'index': index, 'fields_added': added, 'conflicts': conflicts}


async def ensure_items_embedding_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the secondary + backbone embedding fields onto an existing items
    mapping.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    body = {
        'properties': {
            'pe_embedding': _knn_field(dim=config.encoder_embedding_dim),
            BACKBONE_EMBEDDING_FIELD: _knn_field(dim=config.backbone_embedding_dim),
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=['pe_embedding', BACKBONE_EMBEDDING_FIELD],
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': ['pe_embedding', BACKBONE_EMBEDDING_FIELD],
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': ['pe_embedding', BACKBONE_EMBEDDING_FIELD],
            'error': msg,
        }


async def ensure_items_region_embedding(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the region-of-interest embedding + clustering fields onto an
    existing items mapping.

    Encoder embedding of the region-of-interest plus its own clustering
    fields (independent of the item ``cluster_*`` on the same doc), used to
    cluster / AHC-refine regions so false-positives and bad boxes surface
    as outliers. Additive ``PUT <index>/_mapping`` — idempotent.

    F-23/D-2: ``knn_vector``, matching the current items mapping
    (``op_items`` has ``index.knn: true``; the earlier plain-``float``
    rationale here assumed ``index.knn`` was disabled, which is no longer
    true). A ``put_mapping`` against an index still carrying the old plain
    ``float`` mapping fails with a recoverable ``illegal_argument_exception``
    (field type can't change in place — see ``_is_recoverable_mapping_conflict``)
    and is logged at info, not error; the type change itself only takes
    effect on a fresh index or a reindex.
    """
    index = config.items_index
    fields = [F.embedding, F.cluster_id, F.cluster_distance, F.cluster_subid]
    body = {
        'properties': {
            F.embedding: _knn_field(dim=config.encoder_embedding_dim),
            F.cluster_id: {'type': 'integer'},
            F.cluster_distance: {'type': 'float'},
            F.cluster_subid: {'type': 'keyword'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}


async def ensure_items_request_id_field(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the ``request_id`` keyword field onto the existing items mapping.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    body = {'properties': {'request_id': {'type': 'keyword'}}}
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=['request_id'],
            acknowledged=ack,
        )
        return {'acknowledged': ack, 'index': index, 'fields_added': ['request_id']}
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': ['request_id'],
            'error': msg,
        }


async def ensure_items_quality_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the primary-subject rank + blur quality fields onto the existing
    items mapping.

    Adds ``crop_area_norm``, ``crop_rank_in_image``, ``blur_lap_var``,
    ``blur_lap_ratio`` and ``blur_full_var``. Additive
    ``PUT <index>/_mapping`` — idempotent. Mirrors the entries in
    :func:`_items_body`; legacy items are populated by backfill scripts.
    """
    index = config.items_index
    body = {
        'properties': {
            'crop_area_norm': {'type': 'float'},
            'crop_rank_in_image': {'type': 'byte'},
            'blur_lap_var': {'type': 'float'},
            'blur_lap_ratio': {'type': 'float'},
            'blur_full_var': {'type': 'float'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=list(body['properties'].keys()),
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': list(body['properties'].keys()),
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': list(body['properties'].keys()),
            'error': msg,
        }


async def ensure_items_class_name_keyword(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """Add a ``.keyword`` subfield to the ``class_name`` field on the items
    index so terms aggregations + sorts work.

    Some legacy index instantiations mapped ``class_name`` as ``text``
    (no subfield). ``text`` fields aren't doc-values backed, so
    aggregations like ``terms`` on bare ``class_name`` fail with
    ``Text fields are not optimised for operations that require
    per-document field data``. OpenSearch ALLOWS adding subfields to an
    existing text field via ``PUT <index>/_mapping`` — no reindex
    required, no breaking change to text-search behaviour on the parent
    field. Idempotent: calling repeatedly after the subfield exists is
    a no-op acknowledged by OpenSearch.
    """
    index = config.items_index
    body = {
        'properties': {
            'class_name': {
                'type': 'text',
                'fields': {'keyword': {'type': 'keyword', 'ignore_above': 256}},
            }
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info(
            'curation_mapping_migration',
            index=index,
            fields=['class_name.keyword'],
            acknowledged=ack,
        )
        return {
            'acknowledged': ack,
            'index': index,
            'fields_added': ['class_name.keyword'],
        }
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {
            'acknowledged': False,
            'index': index,
            'fields_added': ['class_name.keyword'],
            'error': msg,
        }


async def ensure_items_score_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the curation-scoring overlay fields onto the existing items
    mapping.

    Adds the ``uniqueness_*`` (k-NN density/typicality), ``mistakenness_*``
    (confident-learning margin), and ``dup_*`` (item-level near-duplicate
    grouping) field families. Every field carries a ``{value, method,
    version, scored_at}``-shaped provenance quad so partial re-scores and
    mixed-version pools are visible via ``GET /curation/scores/coverage``.

    Purely additive — these fields are written **only** by the dedicated
    scoring job (``src/services/curation/item_scores/``) and never by the
    production clustering pipeline. No overlay/scorer writes ``cluster_id``
    / ``cluster_subid`` / ``cluster_distance``.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    fields = [
        'uniqueness_score',
        'uniqueness_method',
        'uniqueness_version',
        'uniqueness_scored_at',
        'mistakenness_score',
        'mistakenness_method',
        'mistakenness_version',
        'mistakenness_scored_at',
        'dup_group_id',
        'dup_group_size',
        'dup_is_representative',
        'dup_threshold',
        'dup_method',
        'dup_scored_at',
    ]
    body = {
        'properties': {
            'uniqueness_score': {'type': 'float'},
            'uniqueness_method': {'type': 'keyword'},
            'uniqueness_version': {'type': 'keyword'},
            'uniqueness_scored_at': {'type': 'date'},
            'mistakenness_score': {'type': 'float'},
            'mistakenness_method': {'type': 'keyword'},
            'mistakenness_version': {'type': 'keyword'},
            'mistakenness_scored_at': {'type': 'date'},
            'dup_group_id': {'type': 'keyword'},
            'dup_group_size': {'type': 'integer'},
            'dup_is_representative': {'type': 'boolean'},
            'dup_threshold': {'type': 'float'},
            'dup_method': {'type': 'keyword'},
            'dup_scored_at': {'type': 'date'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}


async def ensure_items_probe_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the probe-model provenance fields onto the existing items
    mapping.

    ``probe_pred_class`` and ``probe_pred_entropy`` are already declared in
    :func:`_items_body`. ``probe_pred_confidence`` and
    ``probe_disagreement`` are written by
    :func:`src.services.curation.probe_predictions.run_probe_inference` —
    this migration declares them explicitly alongside
    ``probe_pred_margin`` (``p(top1) - p(top2)`` from the real per-class
    posterior) and provenance (``probe_model_version`` /
    ``probe_scored_at``).

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    fields = [
        'probe_pred_class_id',
        'probe_pred_confidence',
        'probe_disagreement',
        'probe_pred_margin',
        'probe_model_version',
        'probe_scored_at',
    ]
    body = {
        'properties': {
            'probe_pred_class_id': {'type': 'integer'},
            'probe_pred_confidence': {'type': 'float'},
            'probe_disagreement': {'type': 'boolean'},
            'probe_pred_margin': {'type': 'float'},
            'probe_model_version': {'type': 'keyword'},
            'probe_scored_at': {'type': 'date'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}


async def ensure_items_viz_fields(
    client: AsyncOpenSearch,
) -> dict[str, Any]:
    """PUT the UMAP-visualization overlay fields onto the existing items
    mapping.

    ``viz_x`` / ``viz_y`` are the cached 2-d coordinates from the
    visualization-only UMAP projection
    (:mod:`src.services.curation.embedding_viz`); ``viz_projection_version``
    stamps which fitted projection produced them so a partial re-fit (e.g.
    a cluster-scoped rebuild that only touches some items) is visible the
    same way the score-fields ``*_version`` fields make partial re-scores
    visible.

    **Purely additive and purely cosmetic** — these three fields are
    written **only** by :mod:`embedding_viz`'s background fit job and are
    never read by any clustering code path.

    Additive ``PUT <index>/_mapping`` — idempotent.
    """
    index = config.items_index
    fields = ['viz_x', 'viz_y', 'viz_projection_version']
    body = {
        'properties': {
            'viz_x': {'type': 'float'},
            'viz_y': {'type': 'float'},
            'viz_projection_version': {'type': 'keyword'},
        }
    }
    try:
        resp = await client.indices.put_mapping(index=index, body=body)
        ack = bool(resp.get('acknowledged', False))
        logger.info('curation_mapping_migration', index=index, fields=fields, acknowledged=ack)
        return {'acknowledged': ack, 'index': index, 'fields_added': fields}
    except Exception as exc:
        msg = str(exc)
        is_field_conflict = _is_recoverable_mapping_conflict(msg)
        log_fn = logger.info if is_field_conflict else logger.error
        log_fn(
            'curation_mapping_migration_failed',
            index=index,
            error=msg,
            recoverable=is_field_conflict,
        )
        return {'acknowledged': False, 'index': index, 'fields_added': fields, 'error': msg}


async def mget_crops(
    client: AsyncOpenSearch,
    crop_ids: list[str],
    *,
    index: str = config.items_index,
    source_includes: list[str] | None = None,
    source_excludes: list[str] | None = None,
    seq_no: bool = False,  # noqa: ARG001 - documents caller intent; mget always returns seq_no/primary_term
    chunk_size: int = 256,
) -> dict[str, dict[str, Any]]:
    """Batched ``_mget`` for the items index.

    Replaces per-item ``await client.get(...)`` loops with a single
    round-trip per chunk of up to ``chunk_size`` ids.

    Args:
        client: AsyncOpenSearch instance.
        crop_ids: list of item ids to fetch. Missing ids are silently
            omitted from the result (no KeyError).
        index: target index. Defaults to ``config.items_index`` to
            preserve existing caller behavior; pass explicitly to mget
            against a different index.
        source_includes: if set, restricts the ``_source`` returned
            (keeps the response small). ``None`` returns the full doc.
        source_excludes: if set, drops these fields from ``_source``
            (e.g. large embedding vectors) while keeping everything
            else. Combined with ``source_includes`` only if both are
            given (OpenSearch honors both on the same ``_source``
            clause).
        seq_no: if True, ensures ``_seq_no``/``_primary_term`` come back
            on each doc for OCC. ``mget`` always returns
            ``_seq_no``/``_primary_term`` at the top level of each
            per-doc response regardless of the ``_source`` filter, so
            this flag exists purely for callers to document intent; no
            special body is required, but we keep the param for API
            stability.
        chunk_size: max ids per ``_mget`` call. OS limits the request
            body size; 256 is a safe ceiling.

    Returns:
        ``{crop_id: doc}`` where ``doc`` is the OpenSearch response
        (``_source`` + ``_seq_no``/``_primary_term``).
    """
    if not crop_ids:
        return {}

    source_clause: Any = None
    if source_includes is not None or source_excludes is not None:
        source_clause = {}
        if source_includes is not None:
            source_clause['includes'] = source_includes
        if source_excludes is not None:
            source_clause['excludes'] = source_excludes

    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(crop_ids), chunk_size):
        chunk = crop_ids[start : start + chunk_size]
        body: dict[str, Any] = {
            'docs': [{'_id': cid, '_index': index} for cid in chunk],
        }
        if source_clause is not None:
            for doc in body['docs']:
                doc['_source'] = source_clause
        resp = await client.mget(body=body)
        for d in resp.get('docs', []):
            if not d.get('found'):
                continue
            out[d['_id']] = d
    return out


async def create_curation_indexes(
    client: AsyncOpenSearch,
    cfg: CurationConfig | None = None,
    force_recreate: bool = False,
) -> dict[str, bool]:
    """Create every curation index idempotently.

    Args:
        client: an ``AsyncOpenSearch`` client (already configured).
        cfg: deployment config to resolve index names against. Defaults
            to the module-level singleton.
        force_recreate: drop + recreate every index. **Destructive** — only use
            during bootstrap or in tests.

    Returns:
        Dict of index name -> creation success.
    """
    active_cfg = cfg or config
    results: dict[str, bool] = {}
    for role, body in INDEX_BODIES.items():
        name = index_name(active_cfg, role)
        results[name] = await _create_one(client, name, body, force_recreate)
    return results


# =============================================================================
# ClassRegistry — append-only registry with snapshots + atomic writes
# =============================================================================


class ClassRegistryError(RuntimeError):
    """Raised on registry validation / mutation failures."""


class ClassRegistry:
    """Append-only class registry with on-disk snapshots.

    Backed by ``class_registry.json`` (path from
    ``CurationConfig.class_registry_path``). All mutating ops write a
    snapshot to ``class_registry.<ISO-timestamp>.json`` in the same
    directory **before** rewriting the canonical file (atomic rename via
    tmp file).

    The registry is the source of truth for class IDs — every YOLO
    ``.txt`` label references these IDs.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else config.class_registry_path
        self._cache: ClassRegistryFile | None = None
        self._cache_mtime: float | None = None

    # ------------------------------------------------------------------ I/O

    def _read_disk(self) -> ClassRegistryFile:
        if not self.path.exists():
            logger.warning('curation_registry_missing', path=str(self.path))
            return ClassRegistryFile()
        with self.path.open('r', encoding='utf-8') as f:
            raw = json.load(f)
        return ClassRegistryFile.model_validate(raw)

    def load(self) -> ClassRegistryFile:
        """Load registry, with mtime-based cache invalidation."""
        if not self.path.exists():
            self._cache = ClassRegistryFile()
            self._cache_mtime = None
            return self._cache
        mtime = self.path.stat().st_mtime
        if self._cache is None or self._cache_mtime != mtime:
            self._cache = self._read_disk()
            self._cache_mtime = mtime
        return self._cache

    def _atomic_write(self, registry: ClassRegistryFile) -> None:
        """Snapshot existing file, then atomic-replace canonical file."""
        registry.updated_at = datetime.now(UTC).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Snapshot existing file (if any) BEFORE we mutate.
        if self.path.exists():
            ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
            snapshot_path = self.path.with_name(f'{self.path.stem}.{ts}.json')
            snapshot_path.write_bytes(self.path.read_bytes())
            logger.info('curation_registry_snapshot', snapshot=str(snapshot_path))

        # 2. Atomic write: tmp → fsync → replace.
        tmp_path = self.path.with_suffix(self.path.suffix + '.tmp')
        payload = registry.model_dump_json(indent=2)
        with tmp_path.open('w', encoding='utf-8') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.path)

        # 3. Refresh cache.
        self._cache = registry
        self._cache_mtime = self.path.stat().st_mtime
        logger.info(
            'curation_registry_written', path=str(self.path), n_classes=len(registry.classes)
        )

    # ------------------------------------------------------------------ ops

    def next_id(self) -> int:
        """Return ``max(existing_id) + 1``. ``0`` when registry empty."""
        reg = self.load()
        if not reg.classes:
            return 0
        return max(c.class_id for c in reg.classes) + 1

    def validate_id(self, class_id: int) -> bool:
        """True iff a non-deprecated class with this ID exists."""
        reg = self.load()
        return any(c.class_id == class_id and not c.deprecated for c in reg.classes)

    def get(self, class_id: int) -> RegistryClassEntry | None:
        for c in self.load().classes:
            if c.class_id == class_id:
                return c
        return None

    def add_class(self, name: str, group: str = 'unknown', notes: str = '') -> int:
        """Append a new class. Refuses duplicate (non-deprecated) names.

        Returns the assigned class_id.
        """
        reg = self.load()
        name_norm = name.strip()
        if not name_norm:
            raise ClassRegistryError('class_name must be non-empty')
        for c in reg.classes:
            if c.class_name == name_norm and not c.deprecated:
                raise ClassRegistryError(f'duplicate class_name {name_norm!r} (id={c.class_id})')

        new_id = (max((c.class_id for c in reg.classes), default=-1)) + 1
        entry = RegistryClassEntry(
            class_id=new_id,
            class_name=name_norm,
            group=group,
            notes=notes,
        )
        reg.classes.append(entry)
        self._atomic_write(reg)
        logger.info(
            'curation_registry_add_class', class_id=new_id, class_name=name_norm, group=group
        )
        return new_id

    def rename_class(self, class_id: int, new_name: str) -> RegistryClassEntry:
        """Rename a class in place. Class IDs are immutable; only the name changes."""
        reg = self.load()
        new_name_norm = new_name.strip()
        if not new_name_norm:
            raise ClassRegistryError('new_name must be non-empty')
        for c in reg.classes:
            if c.class_name == new_name_norm and not c.deprecated and c.class_id != class_id:
                raise ClassRegistryError(
                    f'rename target {new_name_norm!r} already in use (id={c.class_id})'
                )
        target: RegistryClassEntry | None = None
        for c in reg.classes:
            if c.class_id == class_id:
                target = c
                old = c.class_name
                c.class_name = new_name_norm
                break
        if target is None:
            raise ClassRegistryError(f'class_id {class_id} not found')
        self._atomic_write(reg)
        logger.info(
            'curation_registry_rename_class',
            class_id=class_id,
            old_name=old,
            new_name=new_name_norm,
        )
        return target

    def merge_class(self, source_id: int, target_id: int) -> dict[str, Any]:
        """Mark ``source_id`` deprecated and record ``merged_into=target_id``.

        Does NOT rewrite labels — bulk relabeling of confirmed labels /
        item docs happens elsewhere.

        Returns:
            ``{source_id, target_id, deprecated, source_name, target_name}``.
        """
        if source_id == target_id:
            raise ClassRegistryError('cannot merge a class into itself')
        reg = self.load()
        source = next((c for c in reg.classes if c.class_id == source_id), None)
        target = next((c for c in reg.classes if c.class_id == target_id), None)
        if source is None:
            raise ClassRegistryError(f'source class_id {source_id} not found')
        if target is None:
            raise ClassRegistryError(f'target class_id {target_id} not found')
        if target.deprecated:
            raise ClassRegistryError(
                f'target class_id {target_id} is deprecated; cannot merge into it'
            )

        source.deprecated = True
        source.merged_into = target_id
        # Note: source.class_id stays burned forever (never re-assigned).
        self._atomic_write(reg)
        logger.info(
            'curation_registry_merge_class',
            source_id=source_id,
            target_id=target_id,
            source_name=source.class_name,
            target_name=target.class_name,
        )
        return {
            'source_id': source_id,
            'target_id': target_id,
            'deprecated': True,
            'source_name': source.class_name,
            'target_name': target.class_name,
        }

    # ------------------------------------------------------------- OS sync

    async def sync_to_opensearch(self, client: AsyncOpenSearch) -> dict[str, int]:
        """Mirror the registry into the classes OpenSearch index.

        Existing docs are upserted by ``class_id``. Deprecated classes remain
        present (with ``deprecated=true``) so dashboards can show history.
        """
        reg = self.load()
        index = index_name(config, IndexRole.CLASSES)

        # Ensure the index exists.
        if not await client.indices.exists(index=index):
            await client.indices.create(index=index, body=INDEX_BODIES[IndexRole.CLASSES])
            logger.info('curation_index_created_on_sync', index=index)

        # F-26: one bulk() instead of one index() per class.
        upserted = 0
        if reg.classes:
            bulk_body: list[dict[str, Any]] = []
            for entry in reg.classes:
                bulk_body.append({'index': {'_index': index, '_id': str(entry.class_id)}})
                bulk_body.append(entry.model_dump())
            resp = await client.bulk(body=bulk_body, refresh=False)
            if isinstance(resp, dict) and resp.get('errors'):
                logger.warning(
                    'curation_registry_sync_partial_errors', sample=resp.get('items', [])[:3]
                )
            upserted = len(reg.classes)
        await client.indices.refresh(index=index)
        logger.info('curation_registry_sync', upserted=upserted, n_classes=len(reg.classes))
        return {'upserted': upserted, 'n_classes': len(reg.classes)}


# =============================================================================
# Module-level convenience
# =============================================================================


_default_registry: ClassRegistry | None = None
_registry_lock = asyncio.Lock()


def get_class_registry() -> ClassRegistry:
    """Return a process-wide :py:class:`ClassRegistry` singleton."""
    global _default_registry  # noqa: PLW0603 - singleton accessor
    if _default_registry is None:
        _default_registry = ClassRegistry()
    return _default_registry


__all__ = [
    'CURATION_SETTINGS_DOC_ID',
    'INDEX_BODIES',
    'ClassRegistry',
    'ClassRegistryError',
    'ClassRegistryFile',
    'RegistryClassEntry',
    'create_curation_indexes',
    'ensure_items_class_name_keyword',
    'ensure_items_embedding_fields',
    'ensure_items_exclusion_fields',
    'ensure_items_history_fields',
    'ensure_items_label_cluster_fields',
    'ensure_items_probe_fields',
    'ensure_items_provenance_fields',
    'ensure_items_quality_fields',
    'ensure_items_region_embedding',
    'ensure_items_request_id_field',
    'ensure_items_score_fields',
    'ensure_items_text_reader_fields',
    'ensure_items_validation_split_fields',
    'ensure_items_viz_fields',
    'ensure_items_vlm_raw_label_fields',
    'get_class_registry',
    'get_curation_index_settings',
    'get_curation_settings',
    'mget_crops',
    'update_curation_settings',
]

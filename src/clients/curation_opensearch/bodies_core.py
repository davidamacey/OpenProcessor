"""Index bodies for the ``images`` and ``items`` roles, and the mapping
fragments the items body and its ``ensure_*`` migrations share."""

from __future__ import annotations

from typing import Any

from src.clients.curation_opensearch.base import F, _knn_field, _knn_settings, config
from src.clients.curation_opensearch.items_extra import ITEMS_EXTRA_MAPPING
from src.config import BACKBONE_EMBEDDING_FIELD
from src.services.curation.item_text import ITEM_TEXT_MAPPING
from src.services.curation.open_vocab_fields import (
    OPEN_VOCAB_IMAGE_MAPPING,
    OPEN_VOCAB_ITEM_MAPPING,
)
from src.services.curation.vlm_class_attempt import VLM_CLASS_ATTEMPT_MAPPING


# Dataset-import provenance (W10.10), on both indexes: the split a frame
# was filed under, the exporter's stem, its stratum string, the hard-negative
# marker, and every import that wrote a label on the doc.
_IMPORT_COMMON_MAPPING: dict[str, Any] = {
    'dataset_split': {'type': 'keyword'},
    'import_ids': {'type': 'keyword'},
    'import_source_stem': {'type': 'keyword'},
    'import_stratum': {'type': 'keyword'},
    'import_hard_negative': {'type': 'boolean'},
}

# Combine provenance (projects plan section 6): where a copied doc came from.
_COMBINE_IMAGES_MAPPING: dict[str, Any] = {
    'origin_project': {'type': 'keyword'},
    'origin_image_id': {'type': 'keyword'},
    'origin_split': {'type': 'keyword'},
}

_COMBINE_ITEMS_MAPPING: dict[str, Any] = {
    **_COMBINE_IMAGES_MAPPING,
    'origin_item_id': {'type': 'keyword'},
    'combine_conflict': {'type': 'boolean'},
    'combine_conflict_origins': {'type': 'keyword'},
    'combine_merged_origins': {'type': 'keyword'},
}

_IMAGES_IMPORT_MAPPING: dict[str, Any] = {
    **_IMPORT_COMMON_MAPPING,
    **_COMBINE_IMAGES_MAPPING,
    # A reviewed negative says "none of THESE classes" (W10.8).
    'import_label_state': {'type': 'keyword'},
    'negative_for': {'type': 'keyword'},
}

_ITEMS_IMPORT_MAPPING: dict[str, Any] = {
    **_IMPORT_COMMON_MAPPING,
    **_COMBINE_ITEMS_MAPPING,
    'imported_at': {'type': 'date'},
    'import_dataset_name': {'type': 'keyword'},
    'import_dataset_sha': {'type': 'keyword'},
    'proposed_by_import': {'type': 'keyword'},
    'on_negative_frame': {'type': 'boolean'},
    'import_standalone_region': {'type': 'boolean'},
    'proposal_chain': {'type': 'keyword'},
}


def _images_body() -> dict[str, Any]:
    return {
        'settings': _knn_settings(),
        'mappings': {
            'properties': {
                'image_id': {'type': 'keyword'},
                'image_path': {'type': 'keyword'},
                # BA-1: the client-supplied identifier for a byte-upload
                # ingest (POST /ingest/upload), kept distinct from
                # image_path once image_path became the server-persisted
                # servable path. null for a server-path ingest.
                'source_identifier': {'type': 'keyword'},
                # BA-4: optional client-supplied tag for one upload call.
                'ingest_run_id': {'type': 'keyword'},
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
                **_IMAGES_IMPORT_MAPPING,
                **OPEN_VOCAB_IMAGE_MAPPING,
            }
        },
    }


# One entry per class write (src/services/curation/history.py). Human label
# writes record the full pre-write class state (class_detector* through
# cluster_subid, restorable=true) so the labeler's Undo restores it exactly.
#
# Mapped as an unindexed object, not `nested`. Nothing ever issues a
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

# Cluster geometry written by the clustering run
# (src/services/curation/clustering/cluster_geometry.py): the cluster id a
# stored cluster_distance was measured against, so a reader can tell a
# distance that went stale when the item moved to another cluster.
CLUSTER_GEOMETRY_MAPPING: dict[str, Any] = {
    'cluster_distance_cluster_id': {'type': 'integer'},
    # Cluster whose centroid is nearest the item (cluster purity, DQ-M2).
    'cluster_nearest_id': {'type': 'integer'},
}

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


def _region_auto_confirm_mapping() -> dict[str, Any]:
    """The worker's auto-confirm flag (``RegionFields.auto_confirmed``): an
    accepted-but-unreviewed region, kept apart from human validation."""
    return {F.auto_confirmed: {'type': 'boolean'}}


def _region_boxes_mapping() -> dict[str, Any]:
    """The W8 multi-box region fields: the box list, the per-box vectors and
    the item-level summary. The one definition behind both the fresh index
    body (:func:`_items_body`) and :func:`ensure_items_region_boxes_fields`."""
    return {
        # W8 multi-box regions: the per-item box list. `nested` so a
        # query like "a box with detector=sam3 AND state=accepted"
        # means the same box (region_boxes.box_query, the only
        # place that builds this nested clause). Element keys are
        # FIXED strings, not RegionFields-indirected (W8.2): the
        # list is new, so no deployment has legacy names for them.
        F.boxes: {
            'type': 'nested',
            'properties': {
                'box_id': {'type': 'keyword'},
                'bbox_norm': {'type': 'float', 'index': False},
                'state': {'type': 'keyword'},
                'score': {'type': 'float'},
                'detector': {'type': 'keyword'},
                'detector_version': {'type': 'keyword'},
                'source': {'type': 'keyword'},
                'bbox_correct': {'type': 'boolean'},
                'confidence': {'type': 'keyword'},
                'rejection_reason': {'type': 'keyword'},
                'text': {'type': 'keyword'},
                'text_raw': {'type': 'keyword'},
                'text_source': {'type': 'keyword'},
                'text_engine_version': {'type': 'keyword'},
                'text_confidence': {'type': 'float'},
                'text_vlm': {'type': 'keyword'},
                'text_ocr': {'type': 'keyword'},
                'text_choice': {'type': 'keyword'},
                'text_vlm_invalid': {'type': 'keyword'},
                'text_disagreement': {'type': 'boolean'},
                'cluster_id': {'type': 'integer'},
                'cluster_subid': {'type': 'keyword'},
                'cluster_distance': {'type': 'float'},
                'detected_at': {'type': 'date'},
            },
        },
        # Per-box vectors live in a SIBLING nested field, not
        # inside F.boxes (W8.2): every item read that feeds
        # serialize_item / every OCC read excludes vectors, and a
        # human edit that read region_boxes with vectors excluded
        # and wrote the list back would silently delete every
        # embedding. Only the embed stage / backfill write this.
        F.box_embeddings: {
            'type': 'nested',
            'properties': {
                'box_id': {'type': 'keyword'},
                # The geometry the vector was computed from: a box
                # moved since then has a stale vector.
                'bbox_norm': {'type': 'float', 'index': False},
                'embedding': _knn_field(dim=config.encoder_embedding_dim),
            },
        },
        F.count: {'type': 'integer'},
        F.rejected_count: {'type': 'integer'},
        F.max_score: {'type': 'float'},
        F.set_complete: {'type': 'boolean'},
        F.revision: {'type': 'integer'},
        F.box_seq: {'type': 'integer', 'index': False},
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
                # Config-store stamps (W2): which activated region profile /
                # revision produced this item's region write, and which
                # prompt pack / revision the VLM used for its most recent
                # write. Null for an item never touched by either. No
                # back-compat migration for a pre-W2 index: stacks are
                # re-created (execution_schedule.md §4.0 NON-NEGOTIABLE 4).
                'region_profile': {'type': 'keyword'},
                'region_profile_revision': {'type': 'integer'},
                'vlm_prompt_pack': {'type': 'keyword'},
                # Which VLM endpoint / model answered the item's most recent
                # VLM write (W9.3): `name@revision` (or `env@<sha12>`) and the
                # resolved model (the probe's `root` when known). Stamped from
                # the answering runtime's identity, so a hot switch never
                # relabels an earlier answer.
                'vlm_endpoint': {'type': 'keyword'},
                'vlm_model': {'type': 'keyword'},
                # VLM's raw answer for every classification call (whether or
                # not it resolved against the registry). Aggregating this field
                # via terms agg surfaces the long-tail labels that should grow
                # the registry to cover. Optional confidence float (0-1) is
                # written when the VLM supplies a numeric score; otherwise the
                # bucketed confidence keyword carries the signal.
                #
                # Field name kept as-is (not indirected via RegionFields —
                # out of scope): this is a live persisted
                # OpenSearch key, and only the region-of-interest
                # fields have an indirection mechanism in Phase 2.
                'vlm_raw_label': {'type': 'keyword'},
                'vlm_raw_label_conf': {'type': 'float'},
                # Marker: class + region resolved in one combined VLM call
                # (scripts/curation/worker/verify.py). Downstream pipeline
                # stages range-query this to skip a redundant class call
                # (src/services/curation/autolabel/selection.py).
                'vlm_verify_completed_at': {'type': 'date'},
                # Last VLM class attempt + why it gave no class
                # (src/services/curation/vlm_class_attempt.py).
                **VLM_CLASS_ATTEMPT_MAPPING,
                # VLM-extracted make/model hint. Field names kept as-is for
                # the same reason as above (no region-of-interest concept
                # applies to this item-level attribute).
                'vlm_item_make': {'type': 'keyword'},
                'vlm_item_model': {'type': 'keyword'},
                # Region-visibility hint: this WAS a vendor- and
                # domain-named field baked into the otherwise-generic
                # index mapping, unlike its siblings above
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
                **CLUSTER_GEOMETRY_MAPPING,
                **ITEMS_EXTRA_MAPPING,
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
                F.verified: {'type': 'boolean'},
                F.reason: {'type': 'text'},
                # Item-level region provenance: the chain of detector steps
                # that ran (a multi-value keyword -- OpenSearch arrays of
                # keyword work as-is), when, and who verified. Per-box
                # geometry / detector / score live in ``region_boxes``.
                F.detector_chain: {'type': 'keyword'},
                F.detected_at: {'type': 'date'},
                F.verifier: {'type': 'keyword'},
                F.verifier_version: {'type': 'keyword'},
                F.verified_at: {'type': 'date'},
                F.rejection_reason: {'type': 'keyword'},
                # Region lifecycle. Explicit so it doesn't fall to dynamic
                # `text` mapping, where terms aggregations and sorts on the
                # bare field name fail.
                F.status: {'type': 'keyword'},
                **_region_auto_confirm_mapping(),
                # Every OCR line read on the item crop + normalized search
                # tokens (src/services/curation/item_text.py).
                **ITEM_TEXT_MAPPING,
                F.class_id: {'type': 'integer'},
                F.label_source: {'type': 'keyword'},
                F.pairing: {'type': 'keyword'},
                F.skip_verify: {'type': 'boolean'},
                F.gate_skip: {'type': 'keyword'},
                # Item label fields written by ingest and the VLM labeler.
                'proposal_name': {'type': 'keyword'},
                # Why an item has no vector (see services/curation/embedding_state.py).
                'embedding_state': {'type': 'keyword'},
                'vlm_confidence': {'type': 'keyword'},
                'vlm_raw_class': {'type': 'keyword'},
                'vlm_proposed_class': {'type': 'keyword'},
                'needs_new_class': {'type': 'boolean'},
                # Quarantine bookkeeping — a legacy-region quarantine script
                # copies the original values into these fields before
                # clearing the live fields and re-running detection.
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
                **_region_boxes_mapping(),
                # History: nested array recording every class write so
                # operators can answer "who labeled this and when" after a
                # model drift investigation. Cap at MAX_HISTORY_ENTRIES (32,
                # see src/services/curation/history.py).
                'class_id_history': _CLASS_HISTORY_MAPPING,
                **_ITEMS_IMPORT_MAPPING,
                # Label Ignore/Undo: exclusion flag + provenance, and the
                # pre-exclusion validation/cluster placement un-exclude
                # restores (src/services/curation/exclusion.py).
                **_EXCLUSION_MAPPING,
                **OPEN_VOCAB_ITEM_MAPPING,
            }
        },
    }

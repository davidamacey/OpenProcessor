"""Shared curation router foundations.

Holds the `router` object, dependency adapters, Pydantic request/response
models, index-name constants, and the `_ensure_indexes` bootstrap. All
sub-modules import from here. _common.py MUST NOT import from sub-modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.clients.curation_opensearch import (
    ClassRegistry,
    create_curation_indexes,
    ensure_items_exclusion_fields,
    ensure_items_history_fields,
    ensure_items_label_cluster_fields,
    ensure_items_pe_v6_embedding_fields,
    ensure_items_probe_fields,
    ensure_items_provenance_fields,
    ensure_items_quality_fields,
    ensure_items_region_embedding,
    ensure_items_request_id_field,
    ensure_items_score_fields,
    ensure_items_text_reader_fields,
    ensure_items_validation_split_fields,
    ensure_items_viz_fields,
    ensure_items_vlm_raw_label_fields,
)
from src.config import IndexRole, get_curation_config, index_name
from src.config.region_state import RegionStatus
from src.core.dependencies import get_opensearch
from src.core.logging import get_logger
from src.services.curation.label_import import DEFAULT_LABEL_SOURCE as _DEFAULT_LABEL_SOURCE


def get_class_registry() -> ClassRegistry:
    """Indirection wrapper so tests patching
    ``src.routers.curation.get_class_registry`` reach every sub-module
    call site. The submodules import this wrapper rather than the
    upstream symbol so the mock applied in ``__init__``'s namespace
    propagates here at call-time.
    """
    from src.routers import curation as _pkg

    return _pkg.get_class_registry()


logger = get_logger(__name__)

config = get_curation_config()


router = APIRouter(
    prefix=config.api_prefix,
    tags=[config.api_tag],
    default_response_class=ORJSONResponse,
)


CURATION_IMAGES_INDEX = index_name(config, IndexRole.IMAGES)
CURATION_ITEMS_INDEX = index_name(config, IndexRole.ITEMS)
CURATION_LABELS_CONFIRMED_INDEX = index_name(config, IndexRole.LABELS_CONFIRMED)
CURATION_CLASSES_INDEX = index_name(config, IndexRole.CLASSES)


_INDEXES_BOOTSTRAPPED = False


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _registry_dep() -> ClassRegistry:
    return get_class_registry()


RegistryDep = Annotated[ClassRegistry, Depends(_registry_dep)]


async def _raw_opensearch_dep() -> Any:
    """Return the raw AsyncOpenSearch instead of the project's
    ``OpenSearchClient`` wrapper."""
    wrapper = await get_opensearch()
    return getattr(wrapper, 'client', wrapper)


OpenSearchDep = Annotated[Any, Depends(_raw_opensearch_dep)]


async def _ensure_indexes(opensearch: Any) -> None:
    """Create curation indexes on first request (idempotent)."""
    global _INDEXES_BOOTSTRAPPED  # noqa: PLW0603 - one-time boot flag
    if _INDEXES_BOOTSTRAPPED:
        return
    try:
        await create_curation_indexes(opensearch, force_recreate=False)
        try:
            await ensure_items_vlm_raw_label_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_vlm_raw_label_migration_failed', error=str(exc))
        try:
            await ensure_items_label_cluster_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_label_cluster_migration_failed', error=str(exc))
        try:
            await ensure_items_provenance_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_provenance_migration_failed', error=str(exc))
        try:
            await ensure_items_request_id_field(opensearch)
        except Exception as exc:
            logger.warning('curation_request_id_migration_failed', error=str(exc))
        # ensure_items_class_name_keyword is NOT called here: the generic
        # `class_name` field is mapped `keyword` directly (not `text`), so
        # this migration's PUT mapping always fails with "cannot be changed
        # from type [keyword] to [text]". All aggregations/scripts query
        # `class_name` directly instead of a `.keyword` subfield that can
        # never exist here. The function itself is left in place for any
        # legacy index that genuinely still maps `class_name` as `text`.
        try:
            await ensure_items_quality_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_quality_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_region_embedding(opensearch)
        except Exception as exc:
            logger.warning('curation_region_embedding_migration_failed', error=str(exc))
        try:
            await ensure_items_score_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_score_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_probe_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_probe_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_viz_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_viz_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_history_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_history_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_exclusion_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_exclusion_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_text_reader_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_text_reader_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_validation_split_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_validation_split_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_pe_v6_embedding_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_pe_v6_embedding_fields_migration_failed', error=str(exc))
        # Self-heal the classes index: if a clean OS wipe left it empty,
        # repopulate from the on-disk class registry so labeling works
        # out of the box. Without this, labeler PUTs fail with
        # "unknown class_id <id>" until an operator manually calls
        # the classes sync endpoint.
        try:
            count_resp = await opensearch.count(index=CURATION_CLASSES_INDEX)
            if (count_resp.get('count') or 0) == 0:
                synced = await get_class_registry().sync_to_opensearch(opensearch)
                logger.info('curation_classes_autosynced', upserted=synced.get('upserted'))
        except Exception as exc:
            logger.warning('curation_classes_autosync_failed', error=str(exc))
        _INDEXES_BOOTSTRAPPED = True
    except Exception as exc:
        logger.warning('curation_index_bootstrap_failed', error=str(exc))


# =============================================================================
# Pydantic models (all request/response models for the curation router live
# here so sub-modules can import them without forming cycles).
# =============================================================================


class IngestImageRequest(BaseModel):
    path: str = Field(..., description='Absolute path to a JPEG on a mounted volume')
    source: str = Field(default='unknown', description='Source tag (e.g. hdd01, dataset_a)')


class IngestImageResponse(BaseModel):
    status: Literal['success', 'duplicate', 'failed']
    image_id: str = ''
    image_path: str
    imohash: str = ''
    n_crops: int = 0
    n_regions: int = 0
    error: str | None = None


class BatchIngestSummaryResponse(BaseModel):
    successful: int = 0
    duplicates: int = 0
    failed: int = 0
    mismatches: int = 0
    missed_labels: int = 0
    unmatched_detections: int = 0
    labels_imported: int = 0
    crops_indexed: int = 0


class BatchIngestResponse(BaseModel):
    status: Literal['success', 'partial', 'error']
    summary: BatchIngestSummaryResponse
    results: list[IngestImageResponse] = Field(default_factory=list)
    # Populated only when the request set ``detect_mismatches``: one record
    # per model-vs-label disagreement (``kind`` = class_mismatch |
    # missed_label | unmatched_detection).
    disagreements: list[dict[str, Any]] = Field(default_factory=list)


class ImportLabelsRequest(BaseModel):
    image_path: str
    label_txt_path: str
    # Defaulted from the label importer so the public API carries no
    # deployment-specific label-source vocabulary.
    label_source: str = _DEFAULT_LABEL_SOURCE
    detect_mismatches: bool = False


class ImportLabelsBatchRequest(BaseModel):
    items: list[ImportLabelsRequest]


class ItemTextLine(BaseModel):
    """One OCR text line on the item crop (``box_norm`` in the item-crop
    frame; ``rel_height`` = line height / crop height)."""

    text: str | None = None
    box_norm: list[float] | None = None
    confidence: float | None = None
    rel_height: float | None = None


class ItemDoc(BaseModel):
    """The wire item every item-returning endpoint emits.

    Documentation/OpenAPI model only: handlers return
    ``src.services.curation.wire.serialize_item`` output directly (a test
    pins this model's fields to that serializer's keys), so a stored value
    of an unexpected type never 500s a browse page. Region attributes use
    the fixed ``region_<attr>`` wire names regardless of any
    ``OP_REGION_FIELD_*`` storage override.
    """

    id: str
    crop_id: str
    image_id: str = ''
    image_path: str = ''
    source_image_path: str = ''
    bbox_norm: list[float] = Field(default_factory=list)
    class_id: int | None = None
    class_name: str | None = ''
    class_source: str | None = ''
    confidence: float = 0.0
    classifier_raw_confidence: float | None = None
    # label_source is nullable: VLM writers set it to None when
    # overwriting a prior validation tag.
    label_source: str | None = ''
    # Derived: class_validated OR region_validated.
    label_validated: bool = False
    class_validated: bool = False
    class_detector: str | None = None
    class_detector_version: str | None = None
    class_labeled_at: str | None = None
    class_labeler: str | None = None
    vlm_confidence: str | None = None
    # VLM class suggestion: the registry class the VLM chose while the
    # label is unvalidated (class_source vlm / vlm_reclassified), or, for
    # vlm_new_class_pending, the proposed new class name with a null id.
    vlm_proposed_class_id: int | None = None
    vlm_proposed_class_name: str | None = None
    cluster_id: int | None = None
    cluster_distance: float | None = None
    # AHC sub-cluster id (e.g. "47a"); cleared whenever cluster_id changes.
    cluster_subid: str | None = None
    test_holdout: bool = False
    crop_rank_in_image: int | None = None
    crop_area_norm: float | None = None
    blur_lap_ratio: float | None = None
    proposal_name: str | None = None
    probe_pred_class: Any = None
    probe_pred_entropy: float | None = None
    mistakenness_score: float | None = None
    mistakenness_method: str | None = None
    mistakenness_version: str | None = None
    mistakenness_scored_at: str | None = None
    uniqueness_score: float | None = None
    dup_group_id: Any = None
    dup_group_size: int | None = None
    dup_is_representative: bool | None = None
    updated_at: str = ''
    thumbnail_url: str = ''
    region_thumbnail_url: str = ''
    region_bbox_norm: list[float] | None = None
    region_bbox_frame: str | None = None
    region_bbox_correct: bool | None = None
    region_status: str | None = None
    region_score: float | None = None
    region_confidence: Any = None
    region_reason: str | None = None
    region_rejection_reason: str | None = None
    region_text: str | None = None
    region_text_raw: str | None = None
    region_text_confidence: float | None = None
    region_text_source: str | None = None
    region_text_engine_version: str | None = None
    region_text_vlm: str | None = None
    region_text_ocr: str | None = None
    region_text_disagreement: bool | None = None
    region_validated: bool | None = None
    region_verified: bool | None = None
    region_verified_at: str | None = None
    region_verifier: str | None = None
    region_verifier_version: str | None = None
    region_visible: bool | None = None
    region_detector: str | None = None
    region_detector_version: str | None = None
    region_detector_chain: list[str] | None = None
    region_detected_at: str | None = None
    region_cluster_id: int | None = None
    region_cluster_subid: str | None = None
    region_cluster_distance: float | None = None
    region_class_id: int | None = None
    region_label_source: str | None = None
    region_source: str | None = None
    region_pairing: Any = None
    region_skip_verify: bool | None = None
    # Every OCR line read on the item crop ([] when none / not yet read).
    item_text_lines: list[ItemTextLine] = Field(default_factory=list)


class CropsPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    crops: list[ItemDoc]
    # Only set for a pool-scale overlay ordering (order='outliers' /
    # 'diverse') — the operator-facing "from N crops in scope" caption
    # needs to know it's looking at a ranked overlay rather than the
    # default newest-first sort. Absent (None) for every other ordering.
    method: str | None = None
    version: str | None = None
    n_pool: int | None = None


class CropLabelRequest(BaseModel):
    class_id: int
    label_source: str = 'human'


class CropBatchLabelRequest(BaseModel):
    crop_ids: list[str]
    class_id: int
    label_source: str = 'human'


class CropMoveRequest(BaseModel):
    crop_ids: list[str]
    cluster_id: int


class CropExcludeRequest(BaseModel):
    """Exclude crops from training + clustering (reversible).

    Blurry / unidentifiable / partial crops the human doesn't want in
    the training set. ``reason`` defaults to ``'ignore'``; the UI can
    pass a more specific tag (``'blurry'``, ``'unidentifiable'``,
    ``'not_a_vehicle'``, ``'partial_crop'``) when the operator wants to
    record why (e.g. a whole cluster of blurry cruisers).
    """

    crop_ids: list[str]
    reason: str = 'ignore'


class CropUnexcludeRequest(BaseModel):
    """Reverse an exclusion (the labeler's Undo path for Ignore)."""

    crop_ids: list[str]


class CropUndoBatchRequest(BaseModel):
    """Undo the most recent human class write on each crop."""

    crop_ids: list[str]


class ItemRegionRequest(BaseModel):
    """Set or clear the region-of-interest sub-bbox on a single item.

    ``region_bbox_norm`` is in the **source-image** coordinate frame; the
    client converts from crop-frame to source-frame before sending.
    ``None`` clears the box and marks the item
    ``region_status='no_region_visible'`` (a deliberate human decision,
    distinct from "not yet detected").
    """

    model_config = {'extra': 'forbid'}

    region_bbox_norm: tuple[float, float, float, float] | None
    region_label_source: str = 'human'


class ItemBatchRegionRequest(BaseModel):
    """Bulk variant of ItemRegionRequest (e.g. "mark these N items as no
    region present")."""

    model_config = {'extra': 'forbid'}

    crop_ids: list[str]
    region_bbox_norm: tuple[float, float, float, float] | None
    region_label_source: str = 'human'


# Whitelist of region status values an operator may write. The detector /
# verify pipeline writes additional values ('pending_detection',
# 'pending_verification', 'detection_failed') that represent transient
# pipeline state — humans never set those by hand.
HUMAN_REGION_STATUS_VALUES = frozenset(
    {
        RegionStatus.DETECTED,
        RegionStatus.NO_REGION_VISIBLE,
        RegionStatus.VERIFY_REJECTED,
        RegionStatus.FALSE_POSITIVE,
    }
)


class CropBatchStatusRequest(BaseModel):
    """Bulk-set ``region_status`` over many items (the cluster-view triage op).

    Lets an operator select an outlier sub-cluster and mark every region
    ``false_positive`` / ``no_region_visible`` in one call, or bulk-confirm
    good regions (``region_status='detected'`` + ``region_verified=True``).
    ``region_status`` must be one of ``HUMAN_REGION_STATUS_VALUES``.
    """

    model_config = {'extra': 'forbid'}

    crop_ids: list[str]
    region_status: str
    region_verified: bool | None = None
    region_label_source: str = 'human'


class ItemRegionMetaRequest(BaseModel):
    """Patch region metadata without touching ``region_bbox_norm``.

    Use this for operator corrections like fixing a region's OCR text or
    changing the status to ``verify_rejected``. To set or clear the bbox
    itself, use ``PUT /crops/{crop_id}/region``.

    Every field is optional; only the provided ones are written. ``None``
    on ``region_text`` clears the text, on ``region_rejection_reason``
    clears the reason. ``region_status`` must be one of
    ``HUMAN_REGION_STATUS_VALUES`` when present.
    """

    # extra='forbid' so a stale key 422s instead of silently no-opping;
    # model_fields_set distinguishes ``region_text=None`` (clear) from
    # "not in payload".
    model_config = {'extra': 'forbid'}

    region_text: str | None = None
    region_status: str | None = None
    region_rejection_reason: str | None = None
    region_label_source: str = 'human'


class ClassEntry(BaseModel):
    class_id: int
    class_name: str
    group: str = ''
    sample_count: int = 0
    validated_count: int = 0
    # FAISS-cluster bucket size: crops whose cluster_id == this class_id.
    # Includes unlabeled candidates that landed near the cluster — i.e.
    # everything visible on /clusters/{id}. The sidebar chip shows this
    # so the operator's eyes match what they'll see when they click in.
    cluster_size: int = 0
    deprecated: bool = False
    # Optional single-character keyboard shortcut. Persisted in the class
    # registry so user customizations survive across sessions and devices.
    # Validated server-side: must be one ASCII char, unique across active
    # classes, not collide with reserved shortcuts.
    hotkey_letter: str | None = None


class ClassListResponse(BaseModel):
    classes: list[ClassEntry]


class ClassCreateRequest(BaseModel):
    name: str
    group: str = 'unknown'
    notes: str = ''


class ClassUpdateRequest(BaseModel):
    name: str | None = None
    group: str | None = None
    # ``""`` clears the binding; ``None`` leaves it unchanged. Single ASCII
    # char only; uniqueness checked server-side at write time.
    hotkey_letter: str | None = None


class ClassMergeRequest(BaseModel):
    source_id: int
    target_id: int


class TestHoldoutFreezeRequest(BaseModel):
    percent: int = Field(default=10, ge=1, le=50)
    # No longer used: selection is deterministic (SHA1-of-crop_id per
    # class, see src/services/curation/test_holdout.py) — a seeded RNG
    # can't guarantee a min-5-per-class floor or reproduce without
    # recording the seed. Kept accepted-but-ignored for backward
    # compatibility with existing callers.
    seed: int = 42


class TestHoldoutFreezeResponse(BaseModel):
    n_frozen: int
    n_classes_covered: int
    test_holdout_sha: str
    per_class_counts: dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded', 'down']
    triton: dict[str, Any]
    opensearch: dict[str, Any]
    vlm: dict[str, Any]
    registry: dict[str, Any]


class ExportYoloRequest(BaseModel):
    export_dir: str | None = None
    # Free-form version tag (e.g. 'v7.0a'). Recorded in manifest.json only.
    version_tag: str = ''
    # RNG seed for the stratified split. Recording it in the manifest is what
    # makes an export re-derivable.
    seed: int = 42
    # Cap distinct source frames collected (representative sample for pipeline
    # tests). None = full export.
    max_images: int | None = None
    # Optional whole-frame near-dup cut (cosine on the images index's
    # secondary embedding). e.g. 0.98 collapses near-identical bursts to
    # one frame; None disables.
    dedup_threshold: float | None = None


class ExportSingleClassRequest(BaseModel):
    """Body for the narrowed single-class / class-subset dataset export.

    Positives come from either the items' own class-labeled boxes
    (``box_source='item'``) or their region-of-interest sub-annotation
    (``box_source='region'``). All ``detected`` positives and all human
    ``false_positive`` hard negatives are kept in full; ``empty_bg_ratio``
    adds a small sample of genuinely empty frames as a fraction of
    positives so the detector still sees some no-target images.
    """

    export_dir: str | None = None
    version_tag: str = ''
    # Registry class ids forming this export's vocabulary, IN ORDER — the
    # dense label id written into the .txt files is the index into this
    # list. Required for box_source='item'; an optional parent-class
    # filter for box_source='region' (empty = every item's region).
    class_ids: list[int] = Field(default_factory=list)
    # 'item' = each item's own bbox_norm; 'region' = its region-of-interest.
    box_source: Literal['item', 'region'] = 'item'
    # data.yaml class name for the single region class in region mode.
    region_class_name: str = 'region'
    # Directory name under the export root; also the manifest's default
    # dataset identity. Keeps this export's artifacts and its own
    # `current` symlink separate from the multi-class export root.
    profile_name: str = 'single_class'
    # RNG seed for the cap sample + the split. Recorded in the manifest,
    # which is what makes an export re-derivable.
    seed: int = 42
    skip_test_split: bool = False
    # Fraction of positives to add as target-free frames (0.1 == 1 per 10).
    empty_bg_ratio: float = 0.1
    # Sample at most this many positive frames; None == all.
    max_positive_images: int | None = None
    # Optional whole-frame near-dup cut (cosine on the images index's
    # secondary embedding). e.g. 0.98 collapses near-identical bursts to
    # one frame; None disables.
    dedup_threshold: float | None = None
    # 'whole_frame' (full source frame) or 'item_crop' (parent item crop
    # with the region re-projected) — for whole-image vs crop training
    # A/B. 'item_crop' requires box_source='region'.
    image_mode: Literal['whole_frame', 'item_crop'] = 'whole_frame'
    # Longest output side in px: 640 for rapid iteration, 1280 for the full run.
    img_max_side: int = 1280
    # False writes labels + artifacts only (fast dry run, no pixel copy).
    copy_images: bool = True


class StatusResponse(BaseModel):
    status: str
    detail: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class _PathLookupRequest(BaseModel):
    """Bulk path-existence check input. Capped client-side to 10k paths."""

    image_paths: list[str] = Field(..., max_length=10_000)


class _PathLookupResponse(BaseModel):
    """Maps known image_path -> existing image_id. Missing paths absent."""

    known_paths: dict[str, str]


class CropFlagNewClassRequest(BaseModel):
    """Marks crops as needing a class that doesn't exist in the registry
    yet — for batch curator review (typically weekly)."""

    crop_ids: list[str]
    note: str = ''


class CurationSettingsResponse(BaseModel):
    """``GET,PUT /curation/settings`` response envelope.

    ``defaults`` is deliberately ``dict[str, str]`` (an OPEN map keyed by
    axis id), not a fixed set of named fields (``cluster``/``sort``/etc.)
    -- a future axis must not require a wire-format change. Missing key =
    no shared override for that axis; the caller falls back to
    ``GET /methods``'s own hardcoded-default resolution (see
    ``src.services.curation.strategy_registry.resolve_effective_default``).
    """

    defaults: dict[str, str] = Field(default_factory=dict)
    updated_at: str | None = None
    updated_by: str | None = None


class CurationSettingsUpdateRequest(BaseModel):
    """``PUT /curation/settings`` body -- partial by design. Only the axes
    present here are validated + merged into the stored document; axes
    already set are left untouched (see
    ``src.clients.curation_opensearch.update_curation_settings``).

    A value of ``null`` for an axis clears that axis's shared override
    (falls back to that endpoint's own hardcoded default / each /review
    tab's own tuned sort) -- otherwise a pinned override was permanently
    unreachable once set, since every id must be currently-advertised and
    there was no way to express "go back to no override" (raised by the
    Cropwright settings-UI integration pass)."""

    defaults: dict[str, str | None] = Field(default_factory=dict)


class _PublishEvent(BaseModel):
    """Event-publish payload — used by out-of-process workers.

    ``extra='forbid'``: a publisher posting an unknown key (e.g. a storage
    field name instead of the wire ``region_status``) gets a 422 instead
    of an event silently stripped of its payload.
    """

    model_config = {'extra': 'forbid'}

    type: str
    crop_id: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    class_source: str | None = None
    region_status: str | None = None
    region_text: str | None = None
    image_path: str | None = None
    topic: str | None = None
    extra: dict[str, Any] | None = None

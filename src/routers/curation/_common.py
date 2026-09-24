"""Shared curation router foundations.

Holds the `router` object, dependency adapters, Pydantic request/response
models, index-name constants, and the `_ensure_indexes` bootstrap. All
sub-modules import from here. _common.py MUST NOT import from sub-modules.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
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
from src.config.region_state import HUMAN_WRITABLE_STATUSES
from src.core.dependencies import get_opensearch
from src.core.logging import get_logger
from src.routers.curation._item_models import CropsPageResponse, ItemDoc  # noqa: F401 - re-export

# Runtime import: pydantic resolves the Literal annotation from module globals.
from src.services.curation.class_sources import HumanLabelSource  # noqa: TC001
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

# OpenSearch's index.max_result_window default. from+size past this 500s
# ("Result window is too large") instead of paging -- reject it explicitly
# with a 422 before it ever reaches OpenSearch (F-7).
MAX_RESULT_WINDOW = 10_000


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def guard_page_depth(page: int, page_size: int) -> None:
    """Raise ``HTTPException(422)`` when ``(page-1)*page_size + page_size``
    would exceed :data:`MAX_RESULT_WINDOW` -- otherwise OpenSearch 500s past
    ``index.max_result_window`` and the app would surface that as a bare
    503/500 instead of a clear, cheap client-side rejection. Cursor-based
    pagination (``search_after``) is the documented way past this limit;
    Wave 1 doesn't add a cursor param, so depth is capped instead."""
    if (page - 1) * page_size + page_size > MAX_RESULT_WINDOW:
        raise HTTPException(
            status_code=422,
            detail=(
                f'page {page} at page_size {page_size} exceeds the {MAX_RESULT_WINDOW} '
                'result-window depth limit; use a smaller page_size or narrow the filter'
            ),
        )


def is_not_found(exc: BaseException) -> bool:
    """A single-doc read failed because the doc doesn't exist (vs. an outage)."""
    return (
        isinstance(exc, KeyError)
        or getattr(exc, 'status_code', None) == 404
        or 'NotFound' in type(exc).__name__
    )


def _registry_dep() -> ClassRegistry:
    return get_class_registry()


RegistryDep = Annotated[ClassRegistry, Depends(_registry_dep)]


async def _raw_opensearch_dep() -> Any:
    """Return the raw AsyncOpenSearch instead of the project's
    ``OpenSearchClient`` wrapper."""
    wrapper = await get_opensearch()
    return getattr(wrapper, 'client', wrapper)


OpenSearchDep = Annotated[Any, Depends(_raw_opensearch_dep)]


async def warm_knn_indexes(opensearch: Any) -> None:
    """Warm the kNN native-engine graph cache for the items + images indexes.

    F-24: without this, the first semantic-search / kNN query after a
    restart (or after a shard relocation) pays the cost of loading the
    faiss/HNSW graph off disk cold. The warmup endpoint forces that load
    to happen once, up front, off the request path.

    Callers should fire this via ``asyncio.create_task`` at startup — it
    must never block app boot, and a failure (endpoint unavailable,
    OpenSearch not up yet, plugin disabled) is logged and swallowed
    rather than raised.
    """
    import time as _time

    indexes = f'{CURATION_ITEMS_INDEX},{CURATION_IMAGES_INDEX}'
    started = _time.monotonic()
    try:
        await opensearch.transport.perform_request('GET', f'/_plugins/_knn/warmup/{indexes}')
    except Exception as exc:
        logger.warning(
            'curation_knn_warmup_failed',
            indexes=indexes,
            duration_s=round(_time.monotonic() - started, 2),
            error=str(exc),
        )
        return
    logger.info(
        'curation_knn_warmup_done',
        indexes=indexes,
        duration_s=round(_time.monotonic() - started, 2),
    )


# F-28.4: guards the whole ~5-exists + N-put_mapping bootstrap sequence
# below. Without this, concurrent requests that all arrive before the
# first one flips _INDEXES_BOOTSTRAPPED each independently race through
# the full migration sequence against OpenSearch (redundant `exists` +
# `put_mapping` calls, all discarded but the first to finish).
_ensure_indexes_lock = asyncio.Lock()


async def _ensure_indexes(opensearch: Any) -> None:
    """Create curation indexes on first request (idempotent).

    Cheap fast path (no lock) once bootstrapped; the lock only guards the
    (at most once) cold-start race.
    """
    if _INDEXES_BOOTSTRAPPED:
        return
    async with _ensure_indexes_lock:
        # Re-check inside the lock: another request may have completed
        # the whole bootstrap sequence while we were waiting to acquire.
        if _INDEXES_BOOTSTRAPPED:
            return
        await _ensure_indexes_locked(opensearch)


async def _ensure_indexes_locked(opensearch: Any) -> None:
    """The actual bootstrap sequence — only ever called while holding
    :data:`_ensure_indexes_lock`. Split out so :func:`_ensure_indexes`'s
    fast path / lock / re-check logic stays readable."""
    global _INDEXES_BOOTSTRAPPED  # noqa: PLW0603 - one-time boot flag
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


class CropLabelRequest(BaseModel):
    class_id: int
    # A human source only; the server writes class_source='human' itself.
    label_source: HumanLabelSource = 'human'


class CropBatchLabelRequest(BaseModel):
    crop_ids: list[str] = Field(..., max_length=5000)
    class_id: int
    label_source: HumanLabelSource = 'human'


class CropMoveRequest(BaseModel):
    crop_ids: list[str] = Field(..., max_length=5000)
    cluster_id: int


class CropExcludeRequest(BaseModel):
    """Exclude crops from training + clustering (reversible).

    Blurry / unidentifiable / partial crops the human doesn't want in
    the training set. ``reason`` defaults to ``'ignore'``; the UI can
    pass a more specific tag (``'blurry'``, ``'unidentifiable'``,
    ``'not_a_vehicle'``, ``'partial_crop'``) when the operator wants to
    record why (e.g. a whole cluster of blurry cruisers).
    """

    crop_ids: list[str] = Field(..., max_length=5000)
    reason: str = 'ignore'


class CropUnexcludeRequest(BaseModel):
    """Reverse an exclusion (the labeler's Undo path for Ignore)."""

    crop_ids: list[str] = Field(..., max_length=5000)


class CropUndoBatchRequest(BaseModel):
    """Undo the most recent human class write on each crop."""

    crop_ids: list[str] = Field(..., max_length=5000)


class CropDiscardRequest(BaseModel):
    """Discard an item: clear its class (it doesn't belong where it is),
    dismiss it from every review queue, or both. Recorded like a label
    write, so ``POST /crops/{id}/label/undo`` reverses it."""

    model_config = {'extra': 'forbid'}

    clear_class: bool = True
    dismiss_from_review: bool = False


class CropDiscardBatchRequest(CropDiscardRequest):
    crop_ids: list[str] = Field(..., max_length=5000)


class ItemRegionRequest(BaseModel):
    """Set or clear the region-of-interest sub-bbox on a single item.

    ``frame`` says which frame ``region_bbox_norm`` is in: ``'source'``
    (the source image, the stored frame) or ``'parent'`` (the item crop;
    the server projects it through the item's own ``bbox_norm``).
    ``None`` clears the box and marks the item
    ``region_status='no_region_visible'`` (a deliberate human decision,
    distinct from "not yet detected").
    """

    model_config = {'extra': 'forbid'}

    region_bbox_norm: tuple[float, float, float, float] | None
    region_label_source: str = 'human'
    frame: Literal['source', 'parent'] = 'source'


class ItemBatchRegionRequest(BaseModel):
    """Bulk variant of ItemRegionRequest (e.g. "mark these N items as no
    region present")."""

    model_config = {'extra': 'forbid'}

    crop_ids: list[str] = Field(..., max_length=5000)
    region_bbox_norm: tuple[float, float, float, float] | None
    region_label_source: str = 'human'
    # 'parent' boxes are projected through each item's own bbox_norm.
    frame: Literal['source', 'parent'] = 'source'


# Region status values an operator may write — the lifecycle entries marked
# human_writable in src/config/region_state.py (stdlib-only, so the TS
# contract codegen exports the same set without importing the app).
HUMAN_REGION_STATUS_VALUES = HUMAN_WRITABLE_STATUSES


class CropBatchStatusRequest(BaseModel):
    """Bulk-set ``region_status`` over many items (the cluster-view triage op).

    Lets an operator select an outlier sub-cluster and mark every region
    ``false_positive`` / ``no_region_visible`` in one call, or bulk-confirm
    good regions (``region_status='detected'``).
    ``region_status`` must be one of ``HUMAN_REGION_STATUS_VALUES``.
    ``region_verified`` is accepted but ignored: the server derives it
    from ``region_status``.
    """

    model_config = {'extra': 'forbid'}

    crop_ids: list[str] = Field(..., max_length=5000)
    region_status: str
    region_verified: bool | None = Field(
        default=None, deprecated=True, description='Ignored; derived from region_status.'
    )
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

    crop_ids: list[str] = Field(..., max_length=5000)
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

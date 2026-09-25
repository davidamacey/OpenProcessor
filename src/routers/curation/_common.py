"""Shared curation router foundations.

Holds the `router` object, dependency adapters, Pydantic request/response
models, index-name constants, and the `_ensure_indexes` bootstrap. All
sub-modules import from here. _common.py MUST NOT import from sub-modules.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import ORJSONResponse

from src.clients.curation_opensearch import (
    ClassRegistry,
    create_curation_indexes,
    ensure_images_upload_fields,
    ensure_items_cluster_geometry_fields,
    ensure_items_embedding_fields,
    ensure_items_exclusion_fields,
    ensure_items_history_fields,
    ensure_items_label_cluster_fields,
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
    ensure_labels_confirmed_fields,
)
from src.config import IndexRole, get_curation_config, index_name
from src.core.dependencies import get_opensearch
from src.core.logging import get_logger
from src.routers.curation._item_models import CropsPageResponse, ItemDoc  # noqa: F401 - re-export


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
# with a 422 before it ever reaches OpenSearch.
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


def _require_region_profile_dep() -> Any:
    """No-profile gating contract: region data/write routes 409 with a
    clear detail when no region profile is configured, rather than
    quietly operating on a region concept that cannot exist yet."""
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    if profile is None:
        raise HTTPException(status_code=409, detail='no region profile is configured')
    return profile


RegionProfileDep = Annotated[Any, Depends(_require_region_profile_dep)]


async def warm_knn_indexes(opensearch: Any) -> None:
    """Warm the kNN native-engine graph cache for the items + images indexes.

    Without this, the first semantic-search / kNN query after a
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


# Guards the whole ~5-exists + N-put_mapping bootstrap sequence
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
        try:
            await ensure_images_upload_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_images_upload_fields_migration_failed', error=str(exc))
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
            await ensure_items_cluster_geometry_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_cluster_geometry_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_text_reader_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_text_reader_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_validation_split_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_validation_split_fields_migration_failed', error=str(exc))
        try:
            await ensure_items_embedding_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_embedding_fields_migration_failed', error=str(exc))
        try:
            await ensure_labels_confirmed_fields(opensearch)
        except Exception as exc:
            logger.warning('curation_labels_confirmed_fields_migration_failed', error=str(exc))
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
# Pydantic models (all request/response models for the curation router) live
# in _common_models.py to stay under the 700-LOC ratchet; re-exported here so
# every existing `from src.routers.curation._common import <Model>` call site
# is unaffected by the split.
# =============================================================================

from src.routers.curation._common_models import (  # noqa: E402
    HUMAN_REGION_STATUS_VALUES,
    BatchIngestResponse,
    BatchIngestSummaryResponse,
    CropBatchLabelRequest,
    CropBatchStatusRequest,
    CropDiscardBatchRequest,
    CropDiscardRequest,
    CropExcludeRequest,
    CropFlagNewClassRequest,
    CropLabelRequest,
    CropMoveRequest,
    CropUndoBatchRequest,
    CropUnexcludeRequest,
    CurationSettingsResponse,
    CurationSettingsUpdateRequest,
    ExportSingleClassRequest,
    ExportYoloRequest,
    HealthResponse,
    ImportLabelsBatchRequest,
    ImportLabelsRequest,
    IngestBatchConfig,
    IngestConfigResponse,
    IngestImageRequest,
    IngestImageResponse,
    IngestRegionDrainConfig,
    IngestRegionDrainResponse,
    IngestStatusResponse,
    IngestUploadConfig,
    ItemBatchRegionRequest,
    ItemRegionMetaRequest,
    ItemRegionRequest,
    RegionDependencyStatusResponse,
    StatusResponse,
    TestHoldoutFreezeRequest,
    TestHoldoutFreezeResponse,
    _PathLookupRequest,
    _PathLookupResponse,
    _PublishEvent,
)


__all__ = [
    'HUMAN_REGION_STATUS_VALUES',
    'BatchIngestResponse',
    'BatchIngestSummaryResponse',
    'CropBatchLabelRequest',
    'CropBatchStatusRequest',
    'CropDiscardBatchRequest',
    'CropDiscardRequest',
    'CropExcludeRequest',
    'CropFlagNewClassRequest',
    'CropLabelRequest',
    'CropMoveRequest',
    'CropUndoBatchRequest',
    'CropUnexcludeRequest',
    'CurationSettingsResponse',
    'CurationSettingsUpdateRequest',
    'ExportSingleClassRequest',
    'ExportYoloRequest',
    'HealthResponse',
    'ImportLabelsBatchRequest',
    'ImportLabelsRequest',
    'IngestBatchConfig',
    'IngestConfigResponse',
    'IngestImageRequest',
    'IngestImageResponse',
    'IngestRegionDrainConfig',
    'IngestRegionDrainResponse',
    'IngestStatusResponse',
    'IngestUploadConfig',
    'ItemBatchRegionRequest',
    'ItemRegionMetaRequest',
    'ItemRegionRequest',
    'RegionDependencyStatusResponse',
    'StatusResponse',
    'TestHoldoutFreezeRequest',
    'TestHoldoutFreezeResponse',
    '_PathLookupRequest',
    '_PathLookupResponse',
    '_PublishEvent',
]

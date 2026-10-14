"""The bound project's curation index bootstrap: create the indexes, run the
mapping migrations and self-heal the classes index, once per project per process.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.ensure_fields import (
    ensure_items_cluster_geometry_fields,
    ensure_items_embedding_fields,
    ensure_items_exclusion_fields,
    ensure_items_history_fields,
    ensure_items_label_cluster_fields,
    ensure_items_provenance_fields,
    ensure_items_region_boxes_fields,
    ensure_items_text_fields,
    ensure_items_validation_split_fields,
    ensure_items_vlm_raw_label_fields,
    ensure_labels_confirmed_fields,
)
from src.clients.curation_opensearch.ensure_overlay_fields import (
    ensure_images_upload_fields,
    ensure_items_inner_result_window,
    ensure_items_probe_fields,
    ensure_items_quality_fields,
    ensure_items_request_id_field,
    ensure_items_score_fields,
    ensure_items_viz_fields,
)
from src.clients.curation_opensearch.items_extra import (
    ensure_items_audit_fields,
    ensure_items_detector_fields,
)
from src.clients.curation_opensearch.lifecycle import create_curation_indexes
from src.clients.curation_opensearch.open_vocab_fields import ensure_open_vocab_fields
from src.clients.curation_opensearch.registry import get_class_registry
from src.config import get_curation_config
from src.config.curation import classes_index, images_index, items_index
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = get_logger(__name__)

# Slugs of the projects whose curation indexes this process has
# bootstrapped (created + migrated). Per project: binding a second
# project must bootstrap its own index set, not skip because another
# project already ran.
_INDEXES_BOOTSTRAPPED: set[str] = set()


async def warm_knn_indexes(opensearch: Any) -> None:
    """Warm the kNN native-engine graph cache for the items + images indexes.

    Without this, the first semantic-search / kNN query after a
    restart (or after a shard relocation) pays the cost of loading the
    faiss/HNSW graph off disk cold. The warmup endpoint forces that load
    to happen once, up front, off the request path. It only loads native
    (faiss) graphs, so it is a no-op for a project created on the Lucene
    engine, whose graph is read with the segment.

    Callers should fire this via ``asyncio.create_task`` at startup — it
    must never block app boot, and a failure (endpoint unavailable,
    OpenSearch not up yet, plugin disabled) is logged and swallowed
    rather than raised.
    """
    import time as _time

    indexes = f'{items_index()},{images_index()}'
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
# first one records their project in _INDEXES_BOOTSTRAPPED each independently race through
# the full migration sequence against OpenSearch (redundant `exists` +
# `put_mapping` calls, all discarded but the first to finish).
_ensure_indexes_lock = asyncio.Lock()


async def ensure_indexes(opensearch: Any) -> None:
    """Create the bound project's curation indexes on its first request
    in this process (idempotent).

    Cheap fast path (no lock) once bootstrapped; the lock only guards the
    (at most once per project) cold-start race.
    """
    slug = get_curation_config().project_slug
    if slug in _INDEXES_BOOTSTRAPPED:
        return
    async with _ensure_indexes_lock:
        # Re-check inside the lock: another request may have completed
        # the whole bootstrap sequence while we were waiting to acquire.
        if slug in _INDEXES_BOOTSTRAPPED:
            return
        await _ensure_indexes_locked(opensearch)


def _migrations() -> tuple[tuple[str, Callable[[Any], Awaitable[Any]]], ...]:
    """The ordered mapping migrations of the bootstrap, looked up at call time."""
    return (
        ('vlm_raw_label', ensure_items_vlm_raw_label_fields),
        ('label_cluster', ensure_items_label_cluster_fields),
        ('provenance', ensure_items_provenance_fields),
        ('request_id', ensure_items_request_id_field),
        ('images_upload_fields', ensure_images_upload_fields),
        ('quality_fields', ensure_items_quality_fields),
        ('region_boxes', ensure_items_region_boxes_fields),
        ('inner_window', ensure_items_inner_result_window),
        ('score_fields', ensure_items_score_fields),
        ('probe_fields', ensure_items_probe_fields),
        ('viz_fields', ensure_items_viz_fields),
        ('history_fields', ensure_items_history_fields),
        ('exclusion_fields', ensure_items_exclusion_fields),
        ('cluster_geometry_fields', ensure_items_cluster_geometry_fields),
        ('audit_fields', ensure_items_audit_fields),
        ('detector_fields', ensure_items_detector_fields),
        ('text_reader_fields', ensure_items_text_fields),
        ('validation_split_fields', ensure_items_validation_split_fields),
        ('embedding_fields', ensure_items_embedding_fields),
        ('open_vocab_fields', ensure_open_vocab_fields),
        ('labels_confirmed_fields', ensure_labels_confirmed_fields),
    )


async def _autosync_classes(opensearch: Any) -> None:
    """Self-heal the classes index: if a clean OS wipe left it empty,
    repopulate from the on-disk class registry so labeling works
    out of the box. Without this, labeler PUTs fail with
    "unknown class_id <id>" until an operator manually calls
    the classes sync endpoint."""
    count_resp = await opensearch.count(index=classes_index())
    if (count_resp.get('count') or 0) == 0:
        synced = await get_class_registry().sync_to_opensearch(opensearch)
        logger.info('curation_classes_autosynced', upserted=synced.get('upserted'))


async def _ensure_indexes_locked(opensearch: Any) -> None:
    """The actual bootstrap sequence — only ever called while holding
    :data:`_ensure_indexes_lock`. A project is recorded as bootstrapped only
    when every step succeeded (an exception, or a migration that reports
    ``acknowledged: false``); otherwise each failure is logged once and the
    next request retries the whole sequence."""
    try:
        await create_curation_indexes(opensearch, force_recreate=False)
    except Exception as exc:
        logger.warning('curation_index_bootstrap_failed', error=str(exc))
        return
    failures: list[str] = []
    for name, migrate in _migrations():
        try:
            result = await migrate(opensearch)
        except Exception as exc:
            logger.warning('curation_migration_failed', migration=name, error=str(exc))
            failures.append(name)
            continue
        if isinstance(result, dict) and result.get('acknowledged') is False:
            logger.warning('curation_migration_failed', migration=name, error=result.get('error'))
            failures.append(name)
    try:
        await _autosync_classes(opensearch)
    except Exception as exc:
        logger.warning('curation_classes_autosync_failed', error=str(exc))
        failures.append('classes_autosync')
    if failures:
        logger.warning('curation_index_bootstrap_incomplete', failed=failures)
        return
    _INDEXES_BOOTSTRAPPED.add(get_curation_config().project_slug)

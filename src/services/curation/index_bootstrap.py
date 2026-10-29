"""The bound project's curation index bootstrap: create the indexes, run the
mapping migrations and self-heal the classes index, once per project per process.

The migrations are additive ``PUT _mapping`` calls, some sixty per project, and every
uvicorn worker bootstraps every active project at startup (#356). So after a full run
each index's mapping ``_meta`` is stamped with :func:`schema_fingerprint`, and a later
process reads every index's marker in one request: when all are current (and the items
index already has its inner-result window) it runs no migration (#376).
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.assist_fields import ensure_assist_fields
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
    adopt_inner_result_window,
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
from src.clients.curation_opensearch.lifecycle import (
    INDEX_BODIES,
    create_curation_indexes,
    curation_index_names,
)
from src.clients.curation_opensearch.open_vocab_fields import ensure_open_vocab_fields
from src.clients.curation_opensearch.registry import get_class_registry
from src.config import get_curation_config
from src.config.curation_accessors import classes_index, images_index, items_index
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = get_logger(__name__)

# Slugs of the projects whose curation indexes this process has
# bootstrapped (created + migrated). Per project: binding a second
# project must bootstrap its own index set, not skip because another
# project already ran.
_INDEXES_BOOTSTRAPPED: set[str] = set()


# The mapping ``_meta`` key holding the schema an index was last fully migrated to.
SCHEMA_META_KEY = 'op_schema'


@functools.cache
def schema_fingerprint() -> str:
    """Names the index schema this code creates and migrates to: a hash of every index
    body and of the migration list. A migration adds only fields its index's body also
    has (``tests/curation/test_index_schema_marker.py`` holds them to it), so adding or
    changing one changes the fingerprint, and every project runs the migrations once more
    on its next bootstrap."""
    payload = {
        'bodies': {role.value: body for role, body in INDEX_BODIES.items()},
        'migrations': [name for name, _ in _migrations()],
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


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


# Guards the whole bootstrap sequence below. Without this, concurrent requests
# that all arrive before the first one records their project in
# _INDEXES_BOOTSTRAPPED each independently race through it against OpenSearch
# (redundant reads, `exists` and `put_mapping` calls, all discarded but the first
# to finish).
_ensure_indexes_lock = asyncio.Lock()


async def ensure_indexes(opensearch: Any) -> bool:
    """Create and migrate the bound project's curation indexes on its first request
    in this process (idempotent); True once the project is bootstrapped.

    Cheap fast path (no lock) once bootstrapped; the lock only guards the
    (at most once per project) cold-start race. A failure is logged and leaves
    the project unbootstrapped, so the next request retries.
    """
    slug = get_curation_config().project_slug
    if slug in _INDEXES_BOOTSTRAPPED:
        return True
    async with _ensure_indexes_lock:
        # Re-check inside the lock: another request may have completed
        # the whole bootstrap sequence while we were waiting to acquire.
        if slug not in _INDEXES_BOOTSTRAPPED:
            await _ensure_indexes_locked(opensearch)
        return slug in _INDEXES_BOOTSTRAPPED


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
        ('assist_fields', ensure_assist_fields),
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


async def _schema_is_current(opensearch: Any, names: list[str]) -> bool:
    """One read of the project's indexes: True when every one exists and carries the
    current schema marker, and the items index already has the inner-result window it
    needs (recorded for the readers, as the migration would). Unreadable is not current."""
    try:
        resp = await opensearch.indices.get(
            index=','.join(names),
            ignore_unavailable=True,
            filter_path=f'*.mappings._meta.{SCHEMA_META_KEY},*.settings.index',
        )
    except Exception as exc:
        logger.warning('curation_schema_marker_read_failed', error=str(exc), exc_info=True)
        return False
    if not isinstance(resp, dict):
        return False
    current = schema_fingerprint()
    for name in names:
        meta = ((resp.get(name) or {}).get('mappings') or {}).get('_meta') or {}
        if meta.get(SCHEMA_META_KEY) != current:
            return False
    items = items_index()
    return adopt_inner_result_window(items, (resp.get(items) or {}).get('settings') or {})


async def _create_and_migrate(opensearch: Any) -> bool:
    """Create the missing indexes and run every migration; False (each failure logged)
    when any step did not succeed: an exception, or a migration that reports
    ``acknowledged: false``."""
    try:
        await create_curation_indexes(opensearch, force_recreate=False)
    except Exception as exc:
        logger.warning('curation_index_bootstrap_failed', error=str(exc))
        return False
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
    if failures:
        logger.warning('curation_index_bootstrap_incomplete', failed=failures)
    return not failures


async def _stamp_schema(opensearch: Any, names: list[str]) -> None:
    """Record on every index that it is migrated to the current schema. Only the next
    process's shortcut depends on it: a failed stamp is logged, and that process runs
    the full bootstrap again."""
    marker = {'_meta': {SCHEMA_META_KEY: schema_fingerprint()}}
    try:
        for name in names:
            await opensearch.indices.put_mapping(index=name, body=marker)
    except Exception as exc:
        logger.warning('curation_schema_marker_write_failed', error=str(exc), exc_info=True)


async def _ensure_indexes_locked(opensearch: Any) -> None:
    """The actual bootstrap sequence — only ever called while holding
    :data:`_ensure_indexes_lock`. A project is recorded as bootstrapped only
    when every step succeeded; otherwise the next request retries the whole
    sequence."""
    names = curation_index_names()
    if not await _schema_is_current(opensearch, names):
        if not await _create_and_migrate(opensearch):
            return
        await _stamp_schema(opensearch, names)
    try:
        await _autosync_classes(opensearch)
    except Exception as exc:
        logger.warning('curation_classes_autosync_failed', error=str(exc))
        logger.warning('curation_index_bootstrap_incomplete', failed=['classes_autosync'])
        return
    _INDEXES_BOOTSTRAPPED.add(get_curation_config().project_slug)

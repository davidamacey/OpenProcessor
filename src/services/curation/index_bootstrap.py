"""The bound project's curation index bootstrap: create the indexes and run the mapping
migrations, then stamp each mapping's ``_meta`` with :func:`schema_fingerprint`.

The migrations are additive ``PUT _mapping`` calls, some sixty per project. They run in the
one-shot ``db-migrate`` step (``scripts/opensearch/migrate.py``) and when a project is created,
never in the API lifespan or a request: the API only checks the stamps
(:func:`check_mappings_current`) and refuses to start on a wrong or missing one.
"""

from __future__ import annotations

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
)
from src.clients.curation_opensearch.ensure_overlay_fields import (
    adopt_inner_result_window,
    ensure_images_upload_fields,
    ensure_items_inner_result_window,
    ensure_items_label_op_field,
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
from src.config import get_curation_config
from src.config.curation_accessors import images_index, items_index
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


class MappingNotCurrentError(RuntimeError):
    """An index of an active project is missing or carries another schema stamp."""

    code = 'opensearch_mapping_not_current'

    def __init__(self, indexes: list[str]):
        super().__init__(
            f'{self.code}: {", ".join(indexes)}; the db-migrate service creates and migrates the '
            'indexes and must complete before the API starts'
        )
        self.indexes = indexes


async def check_mappings_current(opensearch: Any, names: list[str]) -> None:
    """Read the schema stamp of every index in ``names`` in one ``_mapping`` call and raise
    :class:`MappingNotCurrentError` naming each index that is missing or stamped otherwise.

    OpenSearch get-mapping with ``ignore_unavailable`` and ``filter_path``:
    https://docs.opensearch.org/latest/api-reference/index-apis/get-mapping/
    """
    if not names:
        return
    resp = await opensearch.indices.get_mapping(
        index=','.join(names),
        ignore_unavailable=True,
        filter_path=f'*.mappings._meta.{SCHEMA_META_KEY}',
    )
    current = schema_fingerprint()
    stale = [
        name
        for name in names
        if (((resp.get(name) or {}).get('mappings') or {}).get('_meta') or {}).get(SCHEMA_META_KEY)
        != current
    ]
    if stale:
        raise MappingNotCurrentError(stale)


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


async def ensure_indexes(opensearch: Any) -> bool:
    """Create and migrate the bound project's curation indexes on its first request
    in this process (idempotent); True once the project is bootstrapped.

    Safe to run from any number of tasks, workers and processes at once, so there is
    deliberately no lock: an in-process lock cannot exclude another process, and every
    step is idempotent. Index creation treats ``resource_already_exists_exception`` as
    success (create-if-absent), the mapping puts only add fields, and the schema marker
    is written last, only after every migration succeeded. A redundant run costs
    requests, not correctness, and once any run has stamped the marker later runs read
    it and stop. A failure is logged and leaves the project unbootstrapped, so the next
    request retries.
    """
    slug = get_curation_config().project_slug
    if slug not in _INDEXES_BOOTSTRAPPED:
        await _bootstrap_indexes(opensearch)
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
        ('label_op_field', ensure_items_label_op_field),
    )


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


async def _bootstrap_indexes(opensearch: Any) -> None:
    """The bootstrap sequence. A project is recorded as bootstrapped only when every
    step succeeded; otherwise the next request retries the whole sequence."""
    names = curation_index_names()
    if not await _schema_is_current(opensearch, names):
        if not await _create_and_migrate(opensearch):
            return
        await _stamp_schema(opensearch, names)
    _INDEXES_BOOTSTRAPPED.add(get_curation_config().project_slug)

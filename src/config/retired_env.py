"""Startup guard for retired env-var names.

Every rename below is a **clean break** -- no aliases are read. A leftover
retired name is not silently ignored (that would look like a working config
that quietly does nothing, e.g. a stale ``SAM3_URL`` turning the segmenter
off with no error): :func:`reject_retired_env`, called at every process entry
point (``src/app/lifespan.py``'s lifespan, ``scripts/curation/worker/__main__.py``,
``scripts/curation/vlm_worker.py``'s ``main()``, the bake-off evaluator's
``scripts/curation/bakeoff/bakeoff_runner.py`` ``main()``), raises
``RuntimeError`` naming the replacement for every retired name still set.

This module necessarily spells the retired names, so it (and its test)
are allowlisted from the naming-leak scan.
"""

from __future__ import annotations

from src.config.env import env_is_set


# Old name -> new name. A value starting with ``removed:`` marks a setting
# with no one-to-one replacement and says what to use instead.
_REMOVED = 'removed:'
_BAKEOFF_TARGET_REMOVED = (
    f'{_REMOVED} bake-offs score every class; restrict with OP_BAKEOFF_PROFILE_CLASS_FILTER'
)
_FEATURE_OUTPUT_REMOVED = f'{_REMOVED} the profile feature_output field was removed'
RETIRED_ENV: dict[str, str] = {
    'OP_TRITON_URL': 'TRITON_URL',
    'OP_IVF_RETRAIN_CHECK_S': (
        f'{_REMOVED} the IVF retrain is a cluster_run started by the periodic cluster refresh'
    ),
    'VLM_URL': 'OP_VLM_URL',
    'GEMMA_URL': 'OP_VLM_URL',
    'OPENWEBUI_BASE_URL': 'OP_VLM_URL',
    'OPENWEBUI_MODEL': 'OP_VLM_MODEL',
    'OPENWEBUI_API_KEY': 'OP_VLM_API_KEY',
    'VLM_IMAGES_PER_CALL': 'OP_VLM_OPEN_IMAGES_PER_CALL',
    'GEMMA_IMAGES_PER_CALL': 'OP_VLM_OPEN_IMAGES_PER_CALL',
    'VLM_HTTPX_MAX_CONNECTIONS': 'OP_VLM_HTTPX_MAX_CONNECTIONS',
    'GEMMA_HTTPX_MAX_CONNECTIONS': 'OP_VLM_HTTPX_MAX_CONNECTIONS',
    'VLM_HTTPX_KEEPALIVE': 'OP_VLM_HTTPX_KEEPALIVE',
    'GEMMA_HTTPX_KEEPALIVE': 'OP_VLM_HTTPX_KEEPALIVE',
    'SAM3_URL': 'OP_SEGMENTER_URL',
    'SAM3_URLS': 'OP_SEGMENTER_URLS',
    'SAM3_HTTPX_MAX_CONNECTIONS': 'OP_SEGMENTER_HTTPX_MAX_CONNECTIONS',
    'SAM3_HTTPX_KEEPALIVE': 'OP_SEGMENTER_HTTPX_KEEPALIVE',
    'SAM3_SKIP_VLM_VERIFY_SCORE': 'OP_SEGMENTER_SKIP_VERIFY_SCORE',
    'SAM3_SKIP_GEMMA_VERIFY_SCORE': 'OP_SEGMENTER_SKIP_VERIFY_SCORE',
    'SAM_WORKER_VLM_CONCURRENCY': 'OP_REGION_WORKER_VLM_CONCURRENCY',
    'SAM_WORKER_GEMMA_CONCURRENCY': 'OP_REGION_WORKER_VLM_CONCURRENCY',
    'SAM_WORKER_VLM_VISIBLE_CONCURRENCY': 'OP_REGION_WORKER_VLM_VISIBLE_CONCURRENCY',
    'SAM_WORKER_GEMMA_VISIBLE_CONCURRENCY': 'OP_REGION_WORKER_VLM_VISIBLE_CONCURRENCY',
    'SAM_WORKER_METRICS_PORT': 'OP_REGION_WORKER_METRICS_PORT',
    'OP_REGION_DETECTION_SAM_TEXT_PROMPT': 'OP_REGION_DETECTION_SEGMENTER_TEXT_PROMPT',
    'GEMMA_CROP_CACHE_DIR': 'OP_CROP_CACHE_DIR',
    # The backbone embedding field and its dual-head export were removed.
    'OP_BACKBONE_EMBEDDING_DIM': f'{_REMOVED} the backbone_embedding field no longer exists',
    # DetectionProfile.feature_output went with the backbone embedding.
    'OP_INGEST_PRIMARY_FEATURE_OUTPUT': _FEATURE_OUTPUT_REMOVED,
    'OP_INGEST_SECONDARY_FEATURE_OUTPUT': _FEATURE_OUTPUT_REMOVED,
    'OP_REGION_DETECTION_FEATURE_OUTPUT': _FEATURE_OUTPUT_REMOVED,
    # Profiles lost the single target class.
    'OP_BAKEOFF_PROFILE_TARGET_CLASS_ID': _BAKEOFF_TARGET_REMOVED,
    'OP_BAKEOFF_PROFILE_TARGET_CLASS_NAME': _BAKEOFF_TARGET_REMOVED,
    # Events moved from a JSONL file bus to the PostgreSQL events table with a Valkey wake-up.
    'OP_EVENT_BUS': f'{_REMOVED} events are rows in PostgreSQL; there is no file bus',
    'OP_EVENT_LOG_MAX_BYTES': f'{_REMOVED} events are rows in PostgreSQL, trimmed after 24 hours',
    'OP_REGION_DRAIN_STATE_DIR': f'{_REMOVED} the drain streak is the region_drain row of project_state',
    # The VLM endpoint registry is a PostgreSQL table, not an OpenSearch index.
    'OP_GLOBAL_CONFIGS_INDEX': f'{_REMOVED} the endpoint registry is stored in PostgreSQL',
    # The project records are PostgreSQL rows, not an OpenSearch index.
    'OP_PROJECTS_INDEX': f'{_REMOVED} projects are rows in PostgreSQL; there is no projects index',
    # A combine is a jobs row with combine_runs beside it, not a directory of job files.
    'OP_COMBINE_JOBS_DIR': f'{_REMOVED} a combine is a row in jobs; there is no job directory',
    # A dataset import is a dataset_imports row and a jobs row, its ledger rows in the database.
    'OP_DATASET_IMPORTS_DIR': f'{_REMOVED} an import is rows in PostgreSQL, not a directory',
}


def reject_retired_env() -> None:
    """Fail loudly if any retired env var (see :data:`RETIRED_ENV`) is set.

    Raises ``RuntimeError`` listing every retired name still set and its
    replacement. A no-op when none are set.
    """
    stale = sorted(name for name in RETIRED_ENV if env_is_set(name))
    if not stale:
        return
    renames = '; '.join(
        f'{name} was {RETIRED_ENV[name]}'
        if RETIRED_ENV[name].startswith(_REMOVED)
        else f'{name} was renamed to {RETIRED_ENV[name]}'
        for name in stale
    )
    msg = f'retired env var(s) set: {renames}'
    raise RuntimeError(msg)

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
    'OP_AUTO_LABEL_STATE_DIR': (
        f'{_REMOVED} auto-label state is the jobs table; a clustering run is a partition row'
    ),
    'OP_IVF_RETRAIN_CHECK_S': (
        f'{_REMOVED} the IVF retrain is a cluster_run started by the periodic cluster refresh'
    ),
    'OP_CLUSTER_REFRESH_GROWTH_THRESHOLD': (
        f'{_REMOVED} the growth rule is the cluster profile refresh.growth_threshold'
    ),
    'OP_CLUSTER_REFRESH_MAX_DEFERRAL_S': (
        f'{_REMOVED} the deferral bound is the cluster profile refresh.max_deferral_s'
    ),
    'OP_CLUSTER_REFRESH_RETRY_BACKOFF_S': (
        f'{_REMOVED} the retry backoff is the cluster profile refresh.retry_backoff_s'
    ),
    'OP_CLUSTER_REFRESH_AUTO_LABEL': (
        f'{_REMOVED} the periodic refresh starts a cluster_run, never the auto-label job'
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
    # Bulk-ingest runs, sources and queues are PostgreSQL rows, not files.
    'OP_BULK_INGEST_DIR': f'{_REMOVED} runs are bulk_ingest_run jobs and queues are PostgreSQL rows',
    # The open_vocab_drain job is exclusive per project, so no drain's pending stamp is ever reclaimed.
    'OP_OPEN_VOCAB_STALE_S': f'{_REMOVED} one exclusive open_vocab_drain job takes every pending image',
    # The VLM endpoint registry is a PostgreSQL table, not an OpenSearch index.
    'OP_GLOBAL_CONFIGS_INDEX': f'{_REMOVED} the endpoint registry is stored in PostgreSQL',
    # The project records are PostgreSQL rows, not an OpenSearch index.
    'OP_PROJECTS_INDEX': f'{_REMOVED} projects are rows in PostgreSQL; there is no projects index',
    # A combine is a jobs row with combine_runs beside it, not a directory of job files.
    'OP_COMBINE_JOBS_DIR': f'{_REMOVED} a combine is a row in jobs; there is no job directory',
    # A dataset import is a dataset_imports row and a jobs row, its ledger rows in the database.
    'OP_DATASET_IMPORTS_DIR': f'{_REMOVED} an import is rows in PostgreSQL, not a directory',
    # The trainer is a gpu_train queue consumer; the API asks the broker, not the Docker socket.
    'OP_GPU_ARBITER_TRAINER_CONTAINER': (
        f'{_REMOVED} the API finds the trainer as a consumer of the gpu_train queue'
    ),
    # A model export is a jobs row on the gpu_export queue, not a task file.
    'EXPORT_TASK_DIR': f'{_REMOVED} model-export tasks are job rows; read them at GET /models/export/{{task_id}}',
    # Reprocess, assist, assist-loop, probe, select, viz and scores runs are jobs rows. The 0.6.0
    # upgrade reads these from .env and the installer then removes them.
    **dict.fromkeys(
        (
            'OP_REPROCESS_JOBS_DIR',
            'OP_ASSIST_JOBS_DIR',
            'OP_ASSIST_LOOP_JOBS_DIR',
            'OP_PROBE_JOBS_DIR',
            'OP_SELECT_JOBS_DIR',
            'OP_VIZ_JOBS_DIR',
            'OP_SCORES_STATE_DIR',
        ),
        f'{_REMOVED} the run is a jobs row; there is no job directory',
    ),
    'OP_BAKEOFF_JOBS_DIR': f'{_REMOVED} bake-offs are bakeoff jobs; there is no job directory',
    # The pause sentinel file is gone: the VLM workers read deployment_settings
    # ('vlm_paused_for_training'), which the GPU arbiter tick and `openprocessor vlm use` write.
    'OP_PAUSE_SENTINEL': f'{_REMOVED} the VLM workers read the GPU claim from the database',
    'OP_WORKER_PAUSE_SENTINEL': f'{_REMOVED} the detection worker reads the GPU claim from the database',
    'OP_PAUSE_SENTINEL_TTL_S': f'{_REMOVED} a stale `vlm use` pause expires inside the database record',
    'OP_CROP_DUP_THRESHOLD': (
        f'{_REMOVED} the bucketed near-duplicate scorer is gone; near duplicates are run-scoped '
        'groups configured by a near_dup_profile'
    ),
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

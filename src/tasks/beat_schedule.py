"""The beat schedule: every periodic task with its interval (02b section 4.1).

Each entry carries ``expires`` equal to its interval, so a backlog behind a stalled consumer is
dropped instead of replayed (Celery periodic tasks guide). ``src/tasks/app.py`` applies the
schedule when the app's configuration loads, never at import. An interval that an operator can set
(``OP_INGEST_DISPATCH_S`` and the like) is read when the schedule is built; ``0`` leaves the entry
out. ``gpu_arbiter`` is present only when ``OP_GPU_ARBITER_CONTAINERS`` is set.
"""

from __future__ import annotations

from typing import Any

from src.config.cluster_profile import DEFAULT_REFRESH_INTERVAL_S, REFRESH_INTERVAL_ENV
from src.config.env import env_float, env_int, env_str


REAPER_INTERVAL_S = 30
INGEST_DISPATCH_DEFAULT_S = 5.0
OPEN_VOCAB_SWEEP_DEFAULT_S = 120
JOB_STATS_INTERVAL_S = 30
VLM_STAGING_SWEEP_INTERVAL_S = 15 * 60
UPLOAD_SWEEP_INTERVAL_S = 60 * 60
NEAR_DUP_RESCORE_INTERVAL_S = 30
CACHE_PRUNE_INTERVAL_S = 60
RELOAD_PROMOTED_INTERVAL_S = 15
SERVING_INVENTORY_INTERVAL_S = 60
NEAR_DUP_FULL_BUILD_INTERVAL_S = 7 * 24 * 60 * 60
GPU_ARBITER_INTERVAL_S = 15
RESOURCE_REACHABILITY_INTERVAL_S = 30


def _every(task: str, seconds: float) -> dict[str, Any]:
    return {'task': task, 'schedule': float(seconds), 'options': {'expires': seconds}}


def build_schedule() -> dict[str, dict[str, Any]]:
    schedule = {'reaper': _every('op.periodic.reaper', REAPER_INTERVAL_S)}
    ingest = env_float('OP_INGEST_DISPATCH_S', INGEST_DISPATCH_DEFAULT_S, minimum=0.0)
    if ingest > 0:
        schedule['ingest_dispatch'] = _every('op.periodic.ingest_dispatch', ingest)
    sweep = env_int('OP_OPEN_VOCAB_SWEEP_S', OPEN_VOCAB_SWEEP_DEFAULT_S, minimum=0)
    if sweep > 0:
        schedule['open_vocab_sweep'] = _every('op.periodic.open_vocab_sweep', sweep)
    schedule['job_stats'] = _every('op.periodic.job_stats', JOB_STATS_INTERVAL_S)
    schedule['vlm_batch_staging_sweep'] = _every(
        'op.periodic.vlm_batch_staging_sweep', VLM_STAGING_SWEEP_INTERVAL_S
    )
    schedule['upload_sweep'] = _every('op.periodic.upload_sweep', UPLOAD_SWEEP_INTERVAL_S)
    schedule['serving_inventory'] = _every(
        'op.periodic.serving_inventory', SERVING_INVENTORY_INTERVAL_S
    )
    schedule['reload_promoted'] = _every('op.periodic.reload_promoted', RELOAD_PROMOTED_INTERVAL_S)
    schedule['cache_prune'] = _every('op.periodic.cache_prune', CACHE_PRUNE_INTERVAL_S)
    schedule['resource_reachability'] = _every(
        'op.periodic.resource_reachability', RESOURCE_REACHABILITY_INTERVAL_S
    )
    refresh = env_int(REFRESH_INTERVAL_ENV, DEFAULT_REFRESH_INTERVAL_S, minimum=0)
    if refresh > 0:
        schedule['cluster_refresh'] = _every('op.periodic.cluster_refresh', refresh)
    backup_hours = env_float('OP_DB_BACKUP_INTERVAL_H', 0.0, minimum=0.0)
    if backup_hours > 0:
        schedule['db_backup'] = _every('op.periodic.db_backup', backup_hours * 3600)
    if env_str('OP_GPU_ARBITER_CONTAINERS').strip():
        # Only with the overlay: ``expires`` is checked by the worker that receives the message,
        # so a queue with no consumer (``arbiter`` without gpu-arbiter-worker) would pile up.
        schedule['gpu_arbiter'] = _every('op.periodic.gpu_arbiter', GPU_ARBITER_INTERVAL_S)
    schedule['near_dup_rescore'] = _every(
        'op.periodic.near_dup_rescore', NEAR_DUP_RESCORE_INTERVAL_S
    )
    schedule['near_dup_core_group'] = _every(
        'op.periodic.near_dup_core_group', NEAR_DUP_RESCORE_INTERVAL_S
    )
    schedule['near_dup_full_build'] = _every(
        'op.periodic.near_dup_full_build', NEAR_DUP_FULL_BUILD_INTERVAL_S
    )
    return schedule


def apply_schedule(source: Any, **_: Any) -> None:
    """``on_after_configure`` receiver: put the schedule into the loading configuration."""
    source['beat_schedule'] = build_schedule()

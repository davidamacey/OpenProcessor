"""The beat schedule: every periodic task with its interval (02b section 4.1).

Each entry carries ``expires`` equal to its interval, so a backlog behind a stalled consumer is
dropped instead of replayed (Celery periodic tasks guide). ``src/tasks/app.py`` applies the
schedule when the app's configuration loads, never at import. An interval that an operator can set
(``OP_INGEST_DISPATCH_S`` and the like) is read when the schedule is built; ``0`` leaves the entry
out.
"""

from __future__ import annotations

from typing import Any

from src.config.env import env_float, env_int


REAPER_INTERVAL_S = 30
INGEST_DISPATCH_DEFAULT_S = 5.0
OPEN_VOCAB_SWEEP_DEFAULT_S = 120
JOB_STATS_INTERVAL_S = 30
VLM_STAGING_SWEEP_INTERVAL_S = 15 * 60
UPLOAD_SWEEP_INTERVAL_S = 60 * 60


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
    return schedule


def apply_schedule(source: Any, **_: Any) -> None:
    """``on_after_configure`` receiver: put the schedule into the loading configuration."""
    source['beat_schedule'] = build_schedule()

"""The Celery app of the API image: ``celery -A src.tasks.app worker|beat``.

Importing this module imports no task handler's heavy dependencies and opens no connection.
Every task module is in ``include=`` now, so a lane only fills its own module.
"""

from __future__ import annotations

from typing import Any

from celery.signals import beat_init, worker_init

from src.config.retired_env import reject_retired_env
from src.tasks.log_context import install_log_context
from src.tasks.runtime.gpu_app import build_app, install_worker_signals


TASK_MODULES = (
    'src.tasks.infra',
    'src.tasks.reaper',
    'src.tasks.jobs_a',
    'src.tasks.jobs_b',
    'src.tasks.jobs_c',
    'src.tasks.trigger',
    'src.tasks.promote',
    'src.tasks.periodic',
    'src.tasks.b4_fanout',
    'src.tasks.b4_project',
    'src.tasks.beat_schedule',
    'src.tasks.clustering',
    'src.tasks.embedding_reindex',
    'src.tasks.near_dup',
    'src.tasks.near_dup_core',
    'src.tasks.source_backfill',
    'src.tasks.region_text',
)

app = build_app(TASK_MODULES)
install_worker_signals()
install_log_context()


@worker_init.connect(weak=False)
def _on_worker_init(**_: Any) -> None:
    reject_retired_env()


@beat_init.connect(weak=False)
def _on_beat_init(**_: Any) -> None:
    reject_retired_env()

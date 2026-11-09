"""The Celery app of the API image: ``celery -A src.tasks.app worker|beat``.

Importing this module imports no task handler's heavy dependencies and opens no connection.
Every task module is in ``include=`` now, so a lane only fills its own module.
"""

from __future__ import annotations

from typing import Any

from celery.signals import beat_init, worker_init, worker_process_shutdown

from src.config.retired_env import reject_retired_env
from src.tasks import beat_lock, queue_hooks, queue_liveness
from src.tasks.beat_schedule import apply_schedule
from src.tasks.runtime.gpu_app import build_app, install_worker_signals
from src.tasks.runtime.log_context import install_log_context
from src.utils.metrics_processes import forget_process


TASK_MODULES = (
    'src.tasks.infra',
    'src.tasks.reaper',
    'src.tasks.jobs_a',
    'src.tasks.jobs_b',
    'src.tasks.jobs_c',
    'src.tasks.trigger',
    'src.tasks.assist_loop',
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
queue_hooks.install()
queue_liveness.install()
app.on_after_configure.connect(apply_schedule, sender=app, weak=False)


def _require_current_schema() -> None:
    # Imported here: importing this module must not import SQLAlchemy (test_no_cuda_import).
    from src.db import schema_check

    schema_check.require_current_schema()


@worker_init.connect(weak=False)
def _on_worker_init(**_: Any) -> None:
    reject_retired_env()
    _require_current_schema()


@beat_init.connect(weak=False)
def _on_beat_init(**_: Any) -> None:
    reject_retired_env()
    _require_current_schema()


beat_init.connect(beat_lock.hold_single_beat, weak=False)


@worker_process_shutdown.connect(weak=False)
def _on_worker_process_shutdown(pid: int, **_: Any) -> None:
    """A pool child is leaving (``--max-tasks-per-child`` or a stop): drop its live-gauge files."""
    forget_process(pid)

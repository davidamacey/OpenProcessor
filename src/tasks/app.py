"""The Celery app of the API image: ``celery -A src.tasks.app worker|beat``.

Importing this module imports no task handler's heavy dependencies and opens no connection.
Every task module is in ``include=`` now, so a lane only fills its own module.
"""

from __future__ import annotations

from typing import Any

from celery.signals import beat_init, worker_init

from src.config.retired_env import reject_retired_env
from src.tasks import queue_hooks
from src.tasks.beat_schedule import apply_schedule
from src.tasks.runtime.gpu_app import build_app, install_worker_signals
from src.tasks.runtime.log_context import install_log_context


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
queue_hooks.install()
app.on_after_configure.connect(apply_schedule, sender=app, weak=False)


def _require_current_schema() -> None:
    """Exit unless the database answers and is at this image's head.

    Unlike the API, a worker or beat never runs on a database it cannot reach or on another
    revision. Celery logs an ``Exception`` raised by a signal receiver and carries on, so the
    refusal is a ``SystemExit`` (https://docs.celeryq.dev/en/stable/userguide/signals.html).
    """
    # Imported here: importing this module must not import SQLAlchemy (test_no_cuda_import).
    from src.db import schema_check

    try:
        schema_check.wait_for_dependencies({'database': schema_check.database_probe})
        schema_check.check_control_plane()
    except (schema_check.DependencyUnavailableError, schema_check.SchemaNotCurrentError) as exc:
        raise SystemExit(str(exc)) from exc


@worker_init.connect(weak=False)
def _on_worker_init(**_: Any) -> None:
    reject_retired_env()
    _require_current_schema()


@beat_init.connect(weak=False)
def _on_beat_init(**_: Any) -> None:
    reject_retired_env()
    _require_current_schema()

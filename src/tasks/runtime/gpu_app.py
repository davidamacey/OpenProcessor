"""The Celery app factory every image uses, and the worker-start checks.

``build_app`` is the one place the configuration of the job plan (02b section 7.1) is written.
The API image's ``src/tasks/app.py`` calls it; the trainer, export and evaluator images call
``make_app``, which adds the worker signals those images need. Nothing here imports torch,
cuml, tritonclient or opens a database connection at import.
"""

from __future__ import annotations

import ssl
import sys
from typing import TYPE_CHECKING, Any

from celery import Celery
from celery.signals import worker_process_init, worker_ready
from kombu import Queue

from src.config.job_kinds import KIND_BY_TASK
from src.config.queue import NON_JOB_QUEUES, QueueSettings, TaskQueues, broker_transport_options
from src.config.task_routes import ROUTES, celery_task_routes


if TYPE_CHECKING:
    from collections.abc import Sequence


APP_NAME = 'openprocessor'
DEFAULT_SOFT_TIME_LIMIT_S = 10800
DEFAULT_TIME_LIMIT_S = 11700


def static_config(include: Sequence[str]) -> dict[str, Any]:
    """Every setting that does not need the environment."""
    return {
        'include': list(include),
        'task_queues': [Queue(name) for name in TaskQueues.ALL],
        'task_create_missing_queues': False,
        'task_routes': celery_task_routes(),
        'task_serializer': 'json',
        'result_serializer': 'json',
        'event_serializer': 'json',
        'accept_content': ['json'],
        'result_backend': None,
        'task_ignore_result': True,
        'task_track_started': False,
        'task_acks_late': True,
        'task_retry_on_worker_lost': True,
        'worker_prefetch_multiplier': 1,
        'broker_transport_options': broker_transport_options(),
        'broker_connection_retry_on_startup': True,
        'broker_connection_max_retries': None,
        'task_soft_time_limit': DEFAULT_SOFT_TIME_LIMIT_S,
        'task_time_limit': DEFAULT_TIME_LIMIT_S,
        'worker_proc_alive_timeout': 30,
        'worker_soft_shutdown_timeout': 30,
        'worker_enable_soft_shutdown_on_idle': True,
        'worker_cancel_long_running_tasks_on_connection_loss': False,
        'worker_send_task_events': True,
        'task_send_sent_event': True,
        'enable_utc': True,
        'timezone': 'UTC',
    }


def _apply_broker(source: Any, **_: Any) -> None:
    """Read the queue URL and password when the configuration loads, not at import."""
    settings = QueueSettings.from_env()
    source['broker_url'] = settings.broker_url
    if settings.uses_tls:
        source['broker_use_ssl'] = {'ssl_cert_reqs': ssl.CERT_REQUIRED}


def build_app(include: Sequence[str]) -> Celery:
    """A Celery app with the plan's configuration. The configuration loads lazily, on first use."""
    app = Celery(APP_NAME, set_as_current=False)
    app.config_from_object(static_config(include))
    app.on_after_configure.connect(_apply_broker, sender=app, weak=False)
    return app


def make_app(include: Sequence[str]) -> Celery:
    """The app of a GPU image: ``include`` lists its handler modules."""
    install_worker_signals()
    return build_app(include)


def dispose_inherited_engines() -> None:
    """In a pool child: drop database engines inherited from the parent, closing nothing.

    ``dispose(close=False)`` abandons the parent's connections without closing the sockets the
    parent still uses ([pooling and fork](https://docs.sqlalchemy.org/en/21/core/pooling.html)).
    No I/O: the docs allow the child a few seconds. A parent that never imported the engine
    module has nothing to drop.
    """
    engine_module = sys.modules.get('src.db.engine')
    if engine_module is not None:
        engine_module.dispose_inherited_engines()


def task_route_problems(app: Celery) -> tuple[list[str], list[str]]:
    """Registered tasks with no route, and routed job tasks with no kind."""
    names = sorted(name for name in app.tasks if not name.startswith('celery.'))
    unrouted = [name for name in names if name not in ROUTES]
    kindless = [
        name
        for name in names
        if name in ROUTES and ROUTES[name] not in NON_JOB_QUEUES and name not in KIND_BY_TASK
    ]
    return unrouted, kindless


def validate_task_routes(app: Celery) -> None:
    """Refuse to serve when a registered task has no route or no kind (it would hit 'celery')."""
    unrouted, kindless = task_route_problems(app)
    if unrouted or kindless:
        raise SystemExit(f'unrouted tasks: {unrouted}; tasks without a job kind: {kindless}')


def _on_worker_ready(sender: Any = None, **_: Any) -> None:
    from celery import current_app

    validate_task_routes(getattr(sender, 'app', None) or current_app)


def _on_worker_process_init(**_: Any) -> None:
    dispose_inherited_engines()


def install_worker_signals() -> None:
    """Connect the pool-child and ready handlers once (``dispatch_uid`` dedupes repeats)."""
    worker_ready.connect(_on_worker_ready, dispatch_uid='op.worker_ready', weak=False)
    worker_process_init.connect(
        _on_worker_process_init, dispatch_uid='op.worker_process_init', weak=False
    )

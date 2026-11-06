"""The producer side of the queue: send a task by name, and the queued marker.

A minimal Celery app built from ``src.config.queue``; it never imports ``src.tasks.app``, so the
API process loads no task module. The queue of a message comes from the route table only, and
``task_id`` is the job id, so a second send of one job is the same message identity.

The ``op:queued:<job_id>`` key is a derived hint, never truth: the ``jobs`` row is the truth. It
shares fate with the broker message (same server), and an unreachable server gives ``UNKNOWN``,
which nothing acts on.
"""

from __future__ import annotations

import ssl
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING

import redis
from celery import Celery
from kombu.exceptions import KombuError

from src.config.queue import QueueSettings, broker_transport_options, declared_queues
from src.config.task_routes import ROUTES, celery_task_routes
from src.tasks.log_context import install_log_context


if TYPE_CHECKING:
    from collections.abc import Callable

MARKER_PREFIX = 'op:queued:'
_PING_TIMEOUT_S = 1.0
_MARKER_TIMEOUT_S = 1.0
_PUBLISH_RETRY = {'max_retries': 1, 'interval_start': 0, 'interval_step': 0.2, 'interval_max': 0.5}
_BROKER_ERRORS = (KombuError, redis.RedisError, OSError)


class QueueUnavailable(Exception):  # noqa: N818 - name fixed by the job plan (02b section 2.2)
    """The broker could not be reached or refused the message; the caller maps it to a 503."""


class UnknownTaskNameError(LookupError):
    """The task name has no route, so no queue; nothing was sent."""


class QueuedState(StrEnum):
    PRESENT = 'present'
    ABSENT = 'absent'
    UNKNOWN = 'unknown'


@cache
def producer_app() -> Celery:
    """The producer's own Celery app. The URL and password load on first use."""
    return _build_producer(None)


def _build_producer(broker_url: str | None) -> Celery:
    """``broker_url`` is for tests (an in-memory broker); production reads the queue settings."""
    install_log_context()
    app = Celery('openprocessor-producer', set_as_current=False)
    app.config_from_object(
        {
            'task_queues': declared_queues(),
            'task_create_missing_queues': False,
            'task_routes': celery_task_routes(),
            'task_serializer': 'json',
            'accept_content': ['json'],
            'result_backend': None,
            'task_ignore_result': True,
            'broker_transport_options': broker_transport_options(),
            'task_publish_retry': True,
            'task_publish_retry_policy': dict(_PUBLISH_RETRY),
            'enable_utc': True,
            'timezone': 'UTC',
        }
    )
    if broker_url is None:
        app.on_after_configure.connect(_apply_broker, sender=app, weak=False)
    else:
        app.conf.broker_url = broker_url
    return app


def _apply_broker(source: object, **_: object) -> None:
    settings = QueueSettings.from_env()
    source['broker_url'] = settings.broker_url  # type: ignore[index]
    if settings.uses_tls:
        source['broker_use_ssl'] = {'ssl_cert_reqs': ssl.CERT_REQUIRED}  # type: ignore[index]


def enqueue(task_name: str, *, job_id: str) -> None:
    """Publish ``task_name(job_id)`` to the queue its route names, with ``task_id = job_id``."""
    if task_name not in ROUTES:
        raise UnknownTaskNameError(
            f'task {task_name!r} has no route; add it to src/config/task_routes'
        )
    try:
        producer_app().send_task(task_name, args=[job_id], task_id=job_id)
    except _BROKER_ERRORS as exc:
        raise QueueUnavailable(f'cannot publish {task_name}: {exc}') from exc


def _client() -> redis.Redis:
    settings = QueueSettings.from_env()
    return redis.Redis.from_url(
        settings.broker_url,
        socket_timeout=_MARKER_TIMEOUT_S,
        socket_connect_timeout=_MARKER_TIMEOUT_S,
    )


def ping() -> bool:
    """Whether the broker answers within about a second. Never raises."""
    try:
        client = redis.Redis.from_url(
            QueueSettings.from_env().broker_url,
            socket_timeout=_PING_TIMEOUT_S,
            socket_connect_timeout=_PING_TIMEOUT_S,
        )
        try:
            return bool(client.ping())
        finally:
            client.close()
    except _BROKER_ERRORS:
        return False


def _with_client(action: Callable[[redis.Redis], object]) -> object:
    client = _client()
    try:
        return action(client)
    finally:
        client.close()


def mark_queued(job_id: str, *, ttl_s: int) -> None:
    """``SET op:queued:<id> 1 EX ttl_s``; the ttl is the kind's max queue wait."""
    try:
        _with_client(lambda c: c.set(MARKER_PREFIX + job_id, '1', ex=ttl_s))
    except _BROKER_ERRORS as exc:
        raise QueueUnavailable(f'cannot set the queued marker: {exc}') from exc


def clear_queued(job_id: str) -> None:
    try:
        _with_client(lambda c: c.delete(MARKER_PREFIX + job_id))
    except _BROKER_ERRORS as exc:
        raise QueueUnavailable(f'cannot clear the queued marker: {exc}') from exc


def queued_state(job_id: str) -> QueuedState:
    """``PRESENT`` or ``ABSENT``; a connection error is ``UNKNOWN`` (never read as absent)."""
    try:
        exists = _with_client(lambda c: c.exists(MARKER_PREFIX + job_id))
    except _BROKER_ERRORS:
        return QueuedState.UNKNOWN
    return QueuedState.PRESENT if exists else QueuedState.ABSENT

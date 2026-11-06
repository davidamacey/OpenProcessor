"""The Valkey wake-up that tells API processes an event row was committed.

``PUBLISH op:events:<project id | global> 1`` carries no data: the listener re-reads the rows
from its cursor (02b section 6.2). A failed publish is logged and ignored, because the
listener's 2 s safety tick catches up (Valkey PUBLISH: https://valkey.io/commands/publish/,
redis-py: https://redis.readthedocs.io/en/stable/advanced_features.html#publish-subscribe).

A leaf module: every image that writes events (API, job workers, trainer) imports it, so it
touches only ``src.config``, ``src.db`` types and redis-py.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import redis

from src.config.env import EnvError
from src.config.queue import QueueSettings


if TYPE_CHECKING:
    import uuid

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = 'op:events:'
GLOBAL_CHANNEL = f'{CHANNEL_PREFIX}global'
_SOCKET_TIMEOUT_S = 1.0

_clients: dict[int, redis.Redis] = {}


def channel_for(project: uuid.UUID | str | None) -> str:
    """The pub/sub channel of a project id, or the global channel for ``None``."""
    return GLOBAL_CHANNEL if project is None else f'{CHANNEL_PREFIX}{project}'


def client_kwargs(settings: QueueSettings) -> dict[str, Any]:
    """Connection settings shared by the sync wake-up and the API's async subscriber."""
    return {'password': settings.password, 'decode_responses': True}


def _client() -> redis.Redis:
    # One client per process; keyed by pid so a forked Celery child builds its own sockets.
    pid = os.getpid()
    client = _clients.get(pid)
    if client is None:
        settings = QueueSettings.from_env()
        client = redis.Redis.from_url(
            settings.url,
            socket_timeout=_SOCKET_TIMEOUT_S,
            socket_connect_timeout=_SOCKET_TIMEOUT_S,
            **client_kwargs(settings),
        )
        _clients[pid] = client
    return client


def wake(project: uuid.UUID | str | None) -> None:
    """Publish the wake-up for ``project`` (``None``: the global stream). Never raises."""
    try:
        _client().publish(channel_for(project), 1)
    except (redis.RedisError, EnvError, OSError) as exc:  # the safety tick delivers the event
        logger.warning('event_wakeup_failed project=%s error=%s', project, exc)

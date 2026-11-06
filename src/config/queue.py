"""Queue names, the broker settings and the one set of broker transport options.

Pure data and parsing: nothing here imports Celery, so ``src.config`` stays light.
The worker app (``src/tasks/app.py``), the producer (``src/clients/task_queue.py``) and
every GPU image's app (``src/tasks/runtime/gpu_app.py``) read the same values from here.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from src.config.env import EnvError, env_str


class TaskQueues:
    """Every Celery queue. A queue is declared here once and routed in ``task_routes``."""

    JOBS = 'jobs'
    PERIODIC = 'periodic'
    ARBITER = 'arbiter'
    OPEN_VOCAB = 'open_vocab'
    AUTOLABEL = 'autolabel'
    ASSIST_LOOP = 'assist_loop'
    GPU_TRAIN = 'gpu_train'
    GPU_EXPORT = 'gpu_export'
    GPU_EVAL = 'gpu_eval'
    CLUSTER = 'cluster'

    ALL: tuple[str, ...] = (
        JOBS,
        PERIODIC,
        ARBITER,
        OPEN_VOCAB,
        AUTOLABEL,
        ASSIST_LOOP,
        GPU_TRAIN,
        GPU_EXPORT,
        GPU_EVAL,
        CLUSTER,
    )


# Queue -> the compose profile whose service consumes it ('core' is always up). A queue whose
# profile is down holds its messages; the reaper fails the job after its max queue wait.
QUEUE_PROFILE: dict[str, str] = {
    TaskQueues.JOBS: 'core',
    TaskQueues.PERIODIC: 'core',
    TaskQueues.ARBITER: 'overlay',
    TaskQueues.OPEN_VOCAB: 'core',
    TaskQueues.AUTOLABEL: 'curation',
    TaskQueues.ASSIST_LOOP: 'curation',
    TaskQueues.GPU_TRAIN: 'training',
    TaskQueues.GPU_EXPORT: 'export',
    TaskQueues.GPU_EVAL: 'evaluator',
    TaskQueues.CLUSTER: 'curation',
}

# Queues whose tasks are not ``jobs`` rows (beat ticks): they have a route and no job kind.
NON_JOB_QUEUES: frozenset[str] = frozenset({TaskQueues.PERIODIC, TaskQueues.ARBITER})

DEFAULT_QUEUE_URL = 'redis://valkey:6379/0'
_SCHEMES = ('redis', 'rediss')

# Redelivery of an unacknowledged message after this long. Kombu restores unacked messages
# from any consumer with *that consumer's* value, so every app must carry the same one.
VISIBILITY_TIMEOUT_S = 21600
SOCKET_TIMEOUT_S = 30.0
SOCKET_CONNECT_TIMEOUT_S = 10.0
HEALTH_CHECK_INTERVAL_S = 25


@dataclass(frozen=True)
class QueueSettings:
    """``OP_QUEUE_URL`` and the password read from ``OP_QUEUE_PASSWORD_FILE``."""

    url: str
    password: str = field(repr=False)

    @classmethod
    def from_env(cls) -> QueueSettings:
        url = env_str('OP_QUEUE_URL', DEFAULT_QUEUE_URL)
        parts = urlsplit(url)
        if parts.scheme not in _SCHEMES:
            raise EnvError(
                'OP_QUEUE_URL',
                url,
                f'scheme must be one of {_SCHEMES} (kombu has no valkey:// transport)',
            )
        if parts.username is not None or parts.password is not None or '@' in parts.netloc:
            raise EnvError('OP_QUEUE_URL', url, 'must not carry a password; use the password file')
        if not parts.hostname:
            raise EnvError('OP_QUEUE_URL', url, 'has no host')
        return cls(url=url, password=_read_password_file('OP_QUEUE_PASSWORD_FILE'))

    @property
    def broker_url(self) -> str:
        parts = urlsplit(self.url)
        return parts._replace(netloc=f':{quote(self.password, safe="")}@{parts.netloc}').geturl()

    @property
    def uses_tls(self) -> bool:
        return urlsplit(self.url).scheme == 'rediss'


def _read_password_file(name: str) -> str:
    path = env_str(name)
    if not path:
        raise EnvError(name, '', 'is required')
    try:
        password = Path(path).read_text(encoding='utf-8').strip()
    except OSError as exc:
        raise EnvError(name, path, f'cannot be read ({exc.strerror})') from None
    if not password:
        raise EnvError(name, path, 'is empty')
    return password


def broker_transport_options() -> dict[str, Any]:
    """The kombu Redis transport options, identical for every app (review R-33).

    ``retry_on_timeout`` stays False so a dead pub/sub socket reaches kombu, which rebuilds its
    consumer connection. Keepalive plus ``TCP_USER_TIMEOUT`` error a half-open socket in about a
    minute; ``socket_timeout`` stays far above kombu's one-second BRPOP wait.
    """
    keepalive = {
        getattr(socket, name): value
        for name, value in (
            ('TCP_KEEPIDLE', 30),
            ('TCP_KEEPINTVL', 10),
            ('TCP_KEEPCNT', 3),
            ('TCP_USER_TIMEOUT', 60_000),
        )
        if hasattr(socket, name)
    }
    return {
        'visibility_timeout': VISIBILITY_TIMEOUT_S,
        'socket_keepalive': True,
        'socket_keepalive_options': keepalive,
        'socket_timeout': SOCKET_TIMEOUT_S,
        'socket_connect_timeout': SOCKET_CONNECT_TIMEOUT_S,
        'health_check_interval': HEALTH_CHECK_INTERVAL_S,
        'retry_on_timeout': False,
    }

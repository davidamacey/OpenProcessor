"""Connection and thread budgets of one API process.

Every total below is a deployment figure; each process takes ``total // workers`` (the database
share also divides by the replica count). The thread and Triton totals are sized for the 32
workers the stack shipped with (64 threads, 4 channels and 64 in-flight calls per process), so a
smaller ``API_WORKERS`` gives each process proportionally more and the whole API never holds more
than the total.

The database total is a deployment figure shared by every API process of every replica, so
each process takes its share. The worker count is the one uvicorn itself reads
(``WEB_CONCURRENCY``); a missing or non-positive value refuses startup rather than guessing a
count that would oversubscribe the database.
"""

from __future__ import annotations

from src.config.database import DatabaseConfigError, DatabaseSettings
from src.config.env import env_int


EXECUTOR_THREADS_TOTAL = 2048
TRITON_CHANNELS_TOTAL = 128
TRITON_INFLIGHT_TOTAL = 2048

_EXECUTOR_THREADS_FLOOR = 4
_TRITON_CHANNELS_FLOOR = 1
_TRITON_INFLIGHT_FLOOR = 8


def api_worker_count() -> int:
    workers = env_int('WEB_CONCURRENCY', 0)
    if workers < 1:
        raise DatabaseConfigError(
            'WEB_CONCURRENCY',
            'must be set to the number of API worker processes (a positive integer); '
            'the database pool share is computed from it',
        )
    return workers


def db_pool_size(settings: DatabaseSettings | None = None) -> int:
    """Connections each API process may hold: ``total // (workers x replicas)``, at least 1."""
    settings = settings or DatabaseSettings.from_env()
    return max(1, settings.api_connections_total // (api_worker_count() * settings.api_replicas))


def executor_threads() -> int:
    """Threads of the shared CPU executor in this process."""
    return max(_EXECUTOR_THREADS_FLOOR, EXECUTOR_THREADS_TOTAL // api_worker_count())


def triton_channels() -> int:
    """gRPC channels (TCP connections) of this process's async Triton pool."""
    return max(_TRITON_CHANNELS_FLOOR, TRITON_CHANNELS_TOTAL // api_worker_count())


def triton_inflight() -> int:
    """Concurrent Triton calls this process may have outstanding."""
    return max(_TRITON_INFLIGHT_FLOOR, TRITON_INFLIGHT_TOTAL // api_worker_count())

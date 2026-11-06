"""Connection budgets of one API process.

The database total is a deployment figure shared by every API process of every replica, so
each process takes its share. The worker count is the one uvicorn itself reads
(``WEB_CONCURRENCY``); a missing or non-positive value refuses startup rather than guessing a
count that would oversubscribe the database.
"""

from __future__ import annotations

from src.config.database import DatabaseConfigError, DatabaseSettings
from src.config.env import env_int


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

"""How long a request waits for a pooled database connection.

SQLAlchemy's pool events (``connect``, ``checkout``; https://docs.sqlalchemy.org/en/20/core/events.html#sqlalchemy.events.PoolEvents)
fire after a connection is in hand, so none of them can see the wait. ``Pool.connect()`` is the
public call that blocks (up to ``pool_timeout``), so it is timed there. Histograms are
multiprocess-safe: prometheus_client sums buckets across the API workers
(https://prometheus.github.io/client_python/multiprocess/).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import Histogram


if TYPE_CHECKING:
    from sqlalchemy.pool import Pool


POOL_WAIT_SECONDS = Histogram(
    'op_db_pool_wait_seconds',
    'Seconds a request waited for a connection from the API pool (a timeout counts too).',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)


def time_checkouts(pool: Pool) -> None:
    """Record the wait of every ``pool.connect()`` on this pool."""
    connect = pool.connect

    def timed() -> object:
        started = time.perf_counter()
        try:
            return connect()
        finally:
            POOL_WAIT_SECONDS.observe(time.perf_counter() - started)

    pool.connect = timed  # type: ignore[method-assign,assignment]

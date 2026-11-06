"""Lazy, per-process database engines.

Nothing here runs at import: an engine is built on first use and keyed by ``os.getpid()``, so a
process forked after an engine exists (a Celery pool child) builds its own instead of sharing
the parent's sockets. ``dispose_inherited_engines`` drops the parent's pools in the child
without closing the parent's connections, as the SQLAlchemy pooling documentation recommends.

Which engine to use: inside a Celery child, everything on the task's event loop uses a
per-task ``NullPool`` async engine (built in the loop, disposed before it ends); everything on
a thread or outside the loop (claim, heartbeat, terminal write) uses the child's sync engine.
In the API process request code uses the async engine, and the 1 s and 5 s polls use their own
one-connection sync engine so they never queue behind requests. An async session is never
handed to a thread and a sync connection is never used from the loop.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from src.config.api_pools import db_pool_size
from src.config.database import DatabaseSettings


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.ext.asyncio import AsyncEngine

DRIVER = 'postgresql+psycopg'
POOL_TIMEOUT_S = 5
POOL_RECYCLE_S = 1800
# A job worker runs batched backfills and bulk inserts that the API role's short limits would cancel.
JOB_STATEMENT_TIMEOUT = '30min'
JOB_LOCK_TIMEOUT = '30s'
CHILD_POOL_SIZE = 2

_lock = threading.Lock()
_engines: dict[tuple[str, int], Any] = {}


def database_url(
    settings: DatabaseSettings,
    role: str = 'app',
    *,
    direct: bool = False,
    application_name: str | None = None,
) -> URL:
    query = {'sslmode': settings.sslmode}
    if application_name:
        query['application_name'] = application_name
    return URL.create(
        DRIVER,
        username=settings.user(role),
        password=settings.password(role),
        host=settings.direct_host if direct else settings.host,
        port=settings.port,
        database=settings.name,
        query=query,
    )


def _connect_args(settings: DatabaseSettings, *, job_limits: bool) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if settings.pooler == 'pgbouncer':
        # Server-side prepared statements do not survive a transaction pooler.
        args['prepare_threshold'] = None
    elif job_limits:
        args['options'] = (
            f'-c statement_timeout={JOB_STATEMENT_TIMEOUT} -c lock_timeout={JOB_LOCK_TIMEOUT}'
        )
    return args


def _pool_options(size: int, *, timeout: int | None = POOL_TIMEOUT_S) -> dict[str, Any]:
    return {
        'pool_size': size,
        'max_overflow': 0,
        'pool_timeout': timeout,
        'pool_pre_ping': True,
        'pool_recycle': POOL_RECYCLE_S,
        'pool_use_lifo': True,
    }


def _cached[T](kind: str, build: Callable[[], T]) -> T:
    key = (kind, os.getpid())
    with _lock:
        if key not in _engines:
            _engines[key] = build()
        engine: T = _engines[key]
        return engine


def get_async_engine() -> AsyncEngine:
    """The API request-path engine: ``max_overflow=0`` and a 5 s wait that the app maps to 503."""

    def build() -> AsyncEngine:
        settings = DatabaseSettings.from_env()
        return create_async_engine(
            database_url(settings, application_name=f'op-api-{os.getpid()}'),
            connect_args=_connect_args(settings, job_limits=False),
            **_pool_options(db_pool_size(settings)),
        )

    return _cached('async', build)


def get_poll_engine() -> Engine:
    """One connection for the API's registry and config polls, apart from the request pool."""

    def build() -> Engine:
        settings = DatabaseSettings.from_env()
        return create_engine(
            database_url(settings, application_name=f'op-poll-{os.getpid()}'),
            connect_args=_connect_args(settings, job_limits=False),
            **_pool_options(1),
        )

    return _cached('poll', build)


def get_sync_engine(application_name: str | None = None) -> Engine:
    """A Celery child's sync engine (claim, heartbeat thread, terminal write): two connections."""

    def build() -> Engine:
        settings = DatabaseSettings.from_env()
        return create_engine(
            database_url(settings, application_name=application_name or f'op-job-{os.getpid()}'),
            connect_args=_connect_args(settings, job_limits=True),
            **_pool_options(CHILD_POOL_SIZE),
        )

    return _cached('sync', build)


def new_task_async_engine(application_name: str | None = None) -> AsyncEngine:
    """A fresh ``NullPool`` async engine for one task's event loop. The caller disposes it."""
    settings = DatabaseSettings.from_env()
    return create_async_engine(
        database_url(settings, application_name=application_name or f'op-job-{os.getpid()}'),
        connect_args=_connect_args(settings, job_limits=True),
        poolclass=pool.NullPool,
        pool_pre_ping=True,
    )


@contextmanager
def direct_connect(role: str = 'app', application_name: str | None = None) -> Iterator[Connection]:
    """One connection straight to the database host, never through a pooler.

    For session advisory locks (the single-beat lock) and one-off checks. The connection is
    not pooled, so closing it ends the session and releases its locks.
    """
    settings = DatabaseSettings.from_env()
    engine = create_engine(
        database_url(
            settings, role, direct=True, application_name=application_name or f'op-{os.getpid()}'
        ),
        poolclass=pool.NullPool,
    )
    try:
        with engine.connect() as connection:
            yield connection
    finally:
        engine.dispose()


def dispose_inherited_engines() -> None:
    """In a forked child: forget every engine built by the parent without closing its sockets."""
    with _lock:
        for (kind, pid), engine in list(_engines.items()):
            if pid == os.getpid():
                continue
            sync_engine = engine.sync_engine if kind == 'async' else engine
            sync_engine.dispose(close=False)
            del _engines[(kind, pid)]


async def dispose_engines() -> None:
    """Close this process's pools (API shutdown)."""
    with _lock:
        mine = [(k, e) for (k, pid), e in _engines.items() if pid == os.getpid()]
        for kind, _ in mine:
            del _engines[(kind, os.getpid())]
    for kind, engine in mine:
        if kind == 'async':
            await engine.dispose()
        else:
            engine.dispose()


__all__ = [
    'direct_connect',
    'dispose_engines',
    'dispose_inherited_engines',
    'get_async_engine',
    'get_poll_engine',
    'get_sync_engine',
    'new_task_async_engine',
]

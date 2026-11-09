"""One beat per database (D-60): the session advisory lock ``BEAT`` on a direct connection.

Celery requires that "only a single scheduler is running for a schedule at a time"
(https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html). A ``beat_init`` receiver takes
``pg_try_advisory_lock(BEAT, 0)``; when it is held beat exits with code 3. The connection stays open
for beat's life and a watcher thread probes it, so a lost connection ends beat (compose restarts it)
and the server frees the lock when the session dies. The connection sets libpq TCP keepalives and a
user timeout, so a half-open connection fails within about 30 s instead of holding the lock for the
kernel's default retransmission time
(https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS). A session lock
needs a direct connection: a transaction pooler would hand the session to another client.

Nothing here runs at import: psycopg and the database settings load when beat starts.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import TYPE_CHECKING, Any

from src.config.advisory_locks import BEAT


if TYPE_CHECKING:
    from collections.abc import Callable

    import psycopg

    from src.config.database import DatabaseSettings

EXIT_CODE = 3
PROBE_INTERVAL_S = 10.0
# libpq: probe an idle connection after 10 s, every 5 s, drop after 3 misses; fail a send that
# is unacknowledged for 30 s.
_KEEPALIVES = {
    'keepalives': '1',
    'keepalives_idle': '10',
    'keepalives_interval': '5',
    'keepalives_count': '3',
    'tcp_user_timeout': '30000',
}


class BeatAlreadyRunningError(RuntimeError):
    """Another beat holds the lock of this database."""


def conninfo(settings: DatabaseSettings) -> str:
    import psycopg.conninfo

    return psycopg.conninfo.make_conninfo(
        host=settings.direct_host,
        port=settings.port,
        dbname=settings.name,
        user=settings.user('app'),
        password=settings.password('app'),
        sslmode=settings.sslmode,
        application_name=f'op-beat-{os.getpid()}',
        **_KEEPALIVES,
    )


def acquire(settings: DatabaseSettings) -> psycopg.Connection[Any]:
    """Take the beat lock on a new direct connection and return it; the caller keeps it open."""
    import psycopg

    conn = psycopg.connect(conninfo(settings), autocommit=True)
    try:
        row = conn.execute('SELECT pg_try_advisory_lock(%s, 0)', (BEAT,)).fetchone()
    except BaseException:
        conn.close()
        raise
    if row is None or not row[0]:
        conn.close()
        raise BeatAlreadyRunningError('another beat holds the beat lock')
    return conn


def check_alive(conn: psycopg.Connection[Any], *, on_lost: Callable[[], None]) -> None:
    """One probe of the lock connection; ``on_lost`` runs when it no longer answers."""
    import psycopg

    try:
        conn.execute('SELECT 1').fetchone()
    except psycopg.Error:
        on_lost()


def _lost() -> None:
    sys.stderr.write('beat_lock_connection_lost: beat exits so the lock is released\n')
    sys.stderr.flush()
    os._exit(1)  # a watcher thread cannot raise SystemExit in the main thread


def _watch(conn: psycopg.Connection[Any], stop: threading.Event) -> None:
    while not stop.wait(PROBE_INTERVAL_S):
        check_alive(conn, on_lost=_lost)


def hold_single_beat(**_: Any) -> None:
    """``beat_init`` receiver: hold the lock for the life of the process, or exit with code 3."""
    from src.config.database import DatabaseSettings

    try:
        conn = acquire(DatabaseSettings.from_env())
    except BeatAlreadyRunningError:
        sys.stderr.write('beat_already_running: another beat holds the lock; exiting\n')
        sys.stderr.flush()
        raise SystemExit(EXIT_CODE) from None
    threading.Thread(
        target=_watch, args=(conn, threading.Event()), name='beat-lock-watch', daemon=True
    ).start()

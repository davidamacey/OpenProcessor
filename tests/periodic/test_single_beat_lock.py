"""One beat per database (D-60): a second beat exits with ``beat_already_running`` and exit code 3.

The lock is the session advisory lock ``BEAT`` on a direct connection that stays open for beat's
life; killing the holder frees it, and the connection carries TCP keepalives and a user timeout so
a half-open connection ends beat instead of leaving a zombie holding the lock
(https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS).
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from src.config.database import DatabaseSettings
from src.tasks import beat_lock


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env')]


@pytest.fixture
def settings() -> DatabaseSettings:
    return DatabaseSettings.from_env()


def test_a_second_beat_exits_with_code_3_and_names_why(settings, capsys) -> None:
    first = beat_lock.acquire(settings)
    try:
        with pytest.raises(SystemExit) as exit_info:
            beat_lock.hold_single_beat()
        assert exit_info.value.code == beat_lock.EXIT_CODE == 3
    finally:
        first.close()
    assert 'beat_already_running' in capsys.readouterr().err


def test_killing_the_first_lets_a_restarted_second_take_the_lock(settings) -> None:
    first = beat_lock.acquire(settings)
    with pytest.raises(beat_lock.BeatAlreadyRunningError):
        beat_lock.acquire(settings)
    first.close()
    second = beat_lock.acquire(settings)
    second.close()


def test_the_lock_connection_carries_keepalives_and_a_user_timeout(settings) -> None:
    info = conninfo_to_dict(beat_lock.conninfo(settings))
    assert info['keepalives'] == '1'
    assert info['keepalives_idle'] == '10'
    assert info['keepalives_interval'] == '5'
    assert info['keepalives_count'] == '3'
    assert info['tcp_user_timeout'] == '30000'
    assert info['host'] == settings.direct_host


def test_a_dead_lock_connection_ends_beat(settings) -> None:
    lost: list[str] = []
    conn = beat_lock.acquire(settings)
    conn.close()
    beat_lock.check_alive(conn, on_lost=lambda: lost.append('exit'))
    assert lost == ['exit']


def test_a_live_lock_connection_is_left_alone(settings) -> None:
    lost: list[str] = []
    conn = beat_lock.acquire(settings)
    try:
        beat_lock.check_alive(conn, on_lost=lambda: lost.append('exit'))
        assert lost == []
        assert isinstance(conn, psycopg.Connection)
    finally:
        conn.close()

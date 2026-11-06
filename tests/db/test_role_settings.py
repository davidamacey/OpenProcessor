"""Session limits are set once per role, not repeated in application code."""

from __future__ import annotations

import psycopg
import pytest

from src.db import engine as db_engine


pytestmark = pytest.mark.db

_SHOW = (
    "SELECT current_setting('idle_in_transaction_session_timeout'), "
    "current_setting('statement_timeout'), current_setting('lock_timeout'), "
    "current_setting('TimeZone')"
)


def test_an_application_session_carries_the_short_limits(pg) -> None:
    with psycopg.connect(pg.conninfo('app')) as conn:
        assert conn.execute(_SHOW).fetchone() == ('1min', '30s', '5s', 'UTC')


def test_an_owner_session_may_run_long_migrations_but_never_waits_forever_for_a_lock(pg) -> None:
    with psycopg.connect(pg.conninfo('owner')) as conn:
        assert conn.execute(_SHOW).fetchone() == ('5min', '0', '1min', 'UTC')


def test_a_job_worker_connection_raises_the_statement_and_lock_limits(pg_env) -> None:
    with db_engine.get_sync_engine().connect() as conn:
        row = conn.exec_driver_sql(_SHOW).one()
    assert tuple(row) == ('1min', '30min', '30s', 'UTC')


def test_a_connection_names_its_process(pg_env) -> None:
    with db_engine.get_poll_engine().connect() as conn:
        name = conn.exec_driver_sql("SELECT current_setting('application_name')").scalar_one()
    assert name.startswith('op-poll-')

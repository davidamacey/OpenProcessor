"""The schema-head check and the database dependency wait."""

from __future__ import annotations

from contextlib import contextmanager

import psycopg
import pytest

from src.db import schema_check
from src.db.schema_check import (
    Clock,
    DependencyUnavailableError,
    SchemaNotCurrentError,
    check_control_plane,
    database_probe,
    wait_for_dependencies,
)


pytestmark = pytest.mark.db


@contextmanager
def _as_super(pg):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        yield conn


def test_a_database_at_head_passes(pg_env) -> None:
    check_control_plane()


def test_the_code_head_is_the_single_bundled_revision() -> None:
    assert schema_check.code_head() == '0001_control_plane'


def test_a_database_at_an_older_revision_refuses_naming_both(pg_env) -> None:
    with _as_super(pg_env) as conn:
        conn.execute("UPDATE alembic_version SET version_num = '0000_older'")
        try:
            with pytest.raises(SchemaNotCurrentError) as caught:
                check_control_plane()
        finally:
            conn.execute("UPDATE alembic_version SET version_num = '0001_control_plane'")
    assert '0000_older' in str(caught.value)
    assert '0001_control_plane' in str(caught.value)
    assert str(caught.value).startswith('schema_not_current')


def test_a_database_that_was_never_migrated_refuses_too(pg_env) -> None:
    with _as_super(pg_env) as conn:
        conn.execute('ALTER TABLE alembic_version RENAME TO alembic_version_hidden')
        try:
            with pytest.raises(SchemaNotCurrentError, match='no revision'):
                check_control_plane()
        finally:
            conn.execute('ALTER TABLE alembic_version_hidden RENAME TO alembic_version')


def test_the_database_probe_passes_when_the_application_role_connects(pg_env) -> None:
    database_probe()


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> Clock:
        def sleep(seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds

        return Clock(sleep=sleep, monotonic=lambda: self.now)


def test_an_unreachable_database_refuses_after_the_budget_naming_database(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_DB_HOST', '127.0.0.1')
    monkeypatch.setenv('OP_DB_PORT', '9')
    fake = _FakeClock()
    with pytest.raises(
        DependencyUnavailableError, match='dependency_unavailable: database'
    ) as caught:
        wait_for_dependencies({'database': database_probe}, clock=fake.clock())
    assert caught.value.names == ['database']
    assert sum(fake.sleeps) >= schema_check.DEPENDENCY_WAIT_BUDGET_S
    assert set(fake.sleeps) == {schema_check.DEPENDENCY_WAIT_INTERVAL_S}  # a fixed backoff


def test_a_missing_password_file_is_a_configuration_error_not_an_outage(
    pg_env, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv('OP_DB_APP_PASSWORD_FILE', str(tmp_path / 'absent'))
    with pytest.raises(ValueError, match='OP_DB_APP_PASSWORD_FILE'):
        wait_for_dependencies({'database': database_probe}, clock=_FakeClock().clock())

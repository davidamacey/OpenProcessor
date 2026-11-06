"""The migration runner: serialized, idempotent, and it leaves the application role without DDL."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import psycopg
import pytest
from fixtures.postgres import PgDatabase, _settings_for
from psycopg import errors, sql

from scripts.db import migrate


pytestmark = pytest.mark.db


@pytest.fixture
def empty_database(pg: PgDatabase, tmp_path):
    """A new, unmigrated database on the module's server."""
    name = f'op_runner_{int(time.monotonic_ns())}'
    with psycopg.connect(pg.server.conninfo(), autocommit=True) as conn:
        conn.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    yield _settings_for(pg.server, name, tmp_path)
    with psycopg.connect(pg.server.conninfo(), autocommit=True) as conn:
        conn.execute(
            sql.SQL('DROP DATABASE IF EXISTS {} WITH (FORCE)').format(sql.Identifier(name))
        )


def _versions(pg: PgDatabase, settings) -> list[str]:
    with psycopg.connect(pg.conninfo('super').replace(pg.settings.name, settings.name)) as conn:
        return [r[0] for r in conn.execute('SELECT version_num FROM alembic_version').fetchall()]


def test_two_runners_at_once_both_succeed_and_the_schema_is_at_head_once(
    pg: PgDatabase, empty_database
) -> None:
    barrier = threading.Barrier(2)

    def run() -> None:
        barrier.wait()
        migrate.upgrade(empty_database, super_user=pg.server.super_user)

    with ThreadPoolExecutor(2) as pool:
        for future in [pool.submit(run), pool.submit(run)]:
            future.result(timeout=120)
    assert _versions(pg, empty_database) == ['0002_research_seams']


def test_a_runner_waits_while_another_holds_the_migration_lock(
    pg: PgDatabase, empty_database
) -> None:
    # Bootstrap the roles and database first, so the holder can connect as the owner.
    migrate.upgrade(empty_database, super_user=pg.server.super_user)
    holder = psycopg.connect(
        psycopg.conninfo.make_conninfo(
            host=pg.server.host,
            port=pg.server.port,
            dbname=empty_database.name,
            user=empty_database.owner_user,
            password=empty_database.password('owner'),
        ),
        autocommit=True,
    )
    holder.execute('SELECT pg_advisory_lock(%s, %s)', migrate.MIGRATE_LOCK)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(migrate.upgrade, empty_database, super_user=pg.server.super_user)
        with pytest.raises(FutureTimeout):
            future.result(timeout=3)
        holder.execute('SELECT pg_advisory_unlock(%s, %s)', migrate.MIGRATE_LOCK)
        future.result(timeout=120)
    holder.close()


def test_a_rerun_is_a_no_op(pg: PgDatabase, empty_database) -> None:
    migrate.upgrade(empty_database, super_user=pg.server.super_user)
    migrate.upgrade(empty_database, super_user=pg.server.super_user)
    assert migrate.current(empty_database) == '0002_research_seams'


def test_the_application_role_has_data_rights_but_no_ddl(pg: PgDatabase) -> None:
    with psycopg.connect(pg.conninfo('app'), autocommit=True) as app:
        app.execute('SELECT count(*) FROM project_slugs').fetchone()
        with pytest.raises(errors.InsufficientPrivilege):
            app.execute('CREATE TABLE intruder (id int)')
        with pytest.raises(errors.InsufficientPrivilege):
            app.execute("INSERT INTO alembic_version VALUES ('x')")
        with pytest.raises(errors.InsufficientPrivilege):
            app.execute('DROP TABLE project_slugs')


def test_the_roles_have_the_attributes_the_isolation_depends_on(pg: PgDatabase) -> None:
    with psycopg.connect(pg.server.conninfo()) as conn:
        rows = {
            r[0]: r[1:]
            for r in conn.execute(
                'SELECT rolname, rolsuper, rolbypassrls FROM pg_roles WHERE rolname IN (%s, %s)',
                (pg.settings.app_user, pg.settings.owner_user),
            )
        }
    assert rows[pg.settings.app_user] == (False, False)
    assert rows[pg.settings.owner_user] == (False, True)

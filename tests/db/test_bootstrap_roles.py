"""What the superuser bootstrap leaves behind (pending the wave run: needs PostgreSQL 18).

02a section 1.5 and 15a section 6.1: the roles, the extensions and the schema ``vec`` are made by
``src/db/roles.py`` as the superuser, because the owner role has no ``CREATE`` on the database; the
migration only checks them; ``op_vector`` makes tables in ``vec``, ``op_app`` only uses them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg
import pytest
from psycopg import errors

from scripts.db import migrate
from src.db import vector_ddl


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture(scope='module')
def catalog(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        yield conn


def test_the_owner_role_has_no_create_on_the_database_so_it_cannot_make_the_schema(
    pg: PgDatabase,
) -> None:
    owner = pg.settings.owner_user
    with psycopg.connect(pg.conninfo('owner'), autocommit=True) as conn:
        assert conn.execute(
            'SELECT has_database_privilege(%s, current_database(), %s)', (owner, 'CREATE')
        ).fetchone() == (False,)
        with pytest.raises(errors.InsufficientPrivilege):
            conn.execute('CREATE SCHEMA intruder')


def test_the_schema_vec_is_owned_by_the_owner_role_and_the_trigram_extension_is_installed(
    pg: PgDatabase, catalog
) -> None:
    assert catalog.execute(
        "SELECT nspowner::regrole::text FROM pg_namespace WHERE nspname = 'vec'"
    ).fetchone() == (pg.settings.owner_user,)
    assert catalog.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'").fetchone()


def test_the_vector_role_cannot_bypass_row_security_and_is_granted_to_nobody(catalog) -> None:
    role = vector_ddl.OWNER_ROLE
    assert catalog.execute(
        'SELECT rolcanlogin, rolsuper, rolbypassrls, rolcreatedb, rolcreaterole '
        'FROM pg_roles WHERE rolname = %s',
        (role,),
    ).fetchone() == (True, False, False, False, False)
    assert catalog.execute(
        'SELECT count(*) FROM pg_auth_members WHERE member = %s::regrole OR roleid = %s::regrole',
        (role, role),
    ).fetchone() == (0,)


def test_the_vector_role_has_its_timeouts_on_the_role(catalog) -> None:
    settings = catalog.execute(
        'SELECT unnest(setconfig) FROM pg_db_role_setting s JOIN pg_roles r ON r.oid = s.setrole '
        'WHERE r.rolname = %s AND s.setdatabase = 0',
        (vector_ddl.OWNER_ROLE,),
    ).fetchall()
    assert {s[0] for s in settings} == {'statement_timeout=0', 'lock_timeout=0'}


def test_the_vector_role_makes_a_table_in_vec_that_references_projects_and_the_app_role_cannot(
    pg: PgDatabase,
) -> None:
    with psycopg.connect(pg.conninfo('vector'), autocommit=True) as vector:
        for statement in vector_ddl.create_statements(
            0, 'probe_space', 8, app_role=pg.settings.app_user
        ):
            vector.execute(statement)
        assert vector.execute(vector_ddl.row_security_flags(0, 'probe_space')).fetchone() == (True,)
        vector.execute(vector_ddl.drop_statement(0, 'probe_space'))
    with (
        psycopg.connect(pg.conninfo('app'), autocommit=True) as app,
        pytest.raises(errors.InsufficientPrivilege),
    ):
        app.execute('CREATE TABLE vec.intruder (id int)')


def test_a_rerun_of_the_bootstrap_changes_nothing(pg: PgDatabase) -> None:
    migrate.upgrade(pg.settings, super_user=pg.server.super_user)
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM pg_namespace WHERE nspname = 'vec'"
        ).fetchone() == (1,)

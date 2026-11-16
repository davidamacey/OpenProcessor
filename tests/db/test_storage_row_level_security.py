"""Row-level security on every new table: a project's transaction sees and changes only its own rows.

Table-driven over the plan (pending the wave run). One seeded row per table and project; the application
role is a plain role. Cross-project insert is refused by the policy's ``WITH CHECK``, which PostgreSQL
enforces "before any other constraints" (https://www.postgresql.org/docs/18/sql-createpolicy.html), so a
row that names the other project fails with 42501 whatever else it lacks, and a row that names its own
project fails with a constraint error: the policy let it through.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import _storage_plan as plan
import psycopg
import pytest
from _research_seed import bind_scope
from _storage_seed import assert_every_table_seeded, scalar, seed_world
from psycopg import errors


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

PROTECTED = sorted(plan.TABLES - plan.UNPROTECTED)
# analysers and claim_kinds are shared catalogs, seeded once; every other table has one row per project
OWNED = PROTECTED


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        a = seed_world(conn, 'storage-a')
        b = seed_world(conn, 'storage-b')
        assert_every_table_seeded(conn)
    return {'a': a['project'], 'b': b['project']}


def _app(pg: PgDatabase, project: uuid.UUID | None):
    conn = psycopg.connect(pg.conninfo('app'))
    bind_scope(conn, 'project', project)
    return conn


@pytest.mark.parametrize('table', OWNED)
def test_a_project_reads_only_its_own_rows(pg, world, table: str) -> None:
    count = f'SELECT count(*) FROM {table}'  # nosec B608  plan names
    mine_sql = f'{count} WHERE project_id = %s'
    with psycopg.connect(pg.conninfo('super')) as conn:
        total = scalar(conn, count)
        mine = scalar(conn, mine_sql, (world['a'],))
    assert total >= 2 * mine > 0
    with _app(pg, world['a']) as conn:
        assert scalar(conn, count) == mine
        assert scalar(conn, mine_sql, (world['b'],)) == 0


@pytest.mark.parametrize('table', OWNED)
def test_a_project_cannot_update_or_delete_another_projects_rows(pg, world, table: str) -> None:
    with _app(pg, world['a']) as conn:
        updated = conn.execute(
            f'UPDATE {table} SET project_id = project_id WHERE project_id = %s',  # nosec B608
            (world['b'],),
        )
        assert updated.rowcount == 0
        deleted = conn.execute(
            f'DELETE FROM {table} WHERE project_id = %s',  # nosec B608
            (world['b'],),
        )
        assert deleted.rowcount == 0
    with psycopg.connect(pg.conninfo('super')) as conn:
        assert (
            scalar(conn, f'SELECT count(*) FROM {table} WHERE project_id = %s', (world['b'],)) > 0
        )  # nosec B608


@pytest.mark.parametrize('table', OWNED)
def test_a_project_cannot_insert_a_row_for_another_project(pg, world, table: str) -> None:
    with _app(pg, world['a']) as conn, pytest.raises(errors.InsufficientPrivilege):
        conn.execute(f'INSERT INTO {table} (project_id) VALUES (%s)', (world['b'],))  # nosec B608


@pytest.mark.parametrize('table', OWNED)
def test_the_policy_lets_a_row_of_the_projects_own_through_to_the_constraints(
    pg, world, table: str
) -> None:
    with _app(pg, world['a']) as conn, pytest.raises(psycopg.errors.IntegrityError):
        conn.execute(f'INSERT INTO {table} (project_id) VALUES (%s)', (world['a'],))  # nosec B608


@pytest.mark.parametrize('table', OWNED)
def test_a_session_with_no_scope_is_refused_loudly(pg, world, table: str) -> None:
    with (
        psycopg.connect(pg.conninfo('app')) as conn,
        pytest.raises(errors.InsufficientPrivilege, match='scope_not_set'),
    ):
        conn.execute(f'SELECT count(*) FROM {table}')  # nosec B608


@pytest.mark.parametrize('table', sorted(plan.UNPROTECTED))
def test_a_catalog_is_readable_by_every_project_and_changed_by_none(pg, world, table: str) -> None:
    with _app(pg, world['a']) as conn:
        assert scalar(conn, f'SELECT count(*) FROM {table}') > 0  # nosec B608

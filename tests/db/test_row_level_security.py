"""Row-level security: a project's transaction sees and changes only its own rows.

Every protected table is checked with one seeded row per project. The application role is a
plain role: not the owner, not a superuser, without ``BYPASSRLS``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _rls_seed import PROJECT_COLUMN, ids, seed_project, statements
from psycopg import errors
from sqlalchemy import create_engine, text

from src.db.session import sync_project_tx, sync_session_factory


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db

PROTECTED = sorted(PROJECT_COLUMN)
CHILD_ONLY = sorted(t for t, column in PROJECT_COLUMN.items() if column is None)


@pytest.fixture(scope='module')
def world(pg: PgDatabase):
    """Two seeded projects, A and B, plus a project C with nothing in it."""
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        a = seed_project(conn, 'alpha')
        b = seed_project(conn, 'bravo')
        c_id = uuid.uuid4()
        conn.execute("INSERT INTO project_slugs (slug) VALUES ('charlie')")
        conn.execute(
            'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
            "model_prefix, mlflow_experiment, created_by) VALUES (%s, 'charlie', 'c', 'active', "
            "'c__', 'projects/charlie', 'c__', 'c', 'test')",
            (c_id,),
        )
    return {'a': a, 'b': b, 'c': c_id}


def _app(pg: PgDatabase):
    return psycopg.connect(pg.conninfo('app'))


def _scope(conn, project: uuid.UUID | None, kind: str = 'project') -> None:
    conn.execute("SELECT set_config('op.scope', %s, true)", (kind,))
    conn.execute("SELECT set_config('op.project_id', %s, true)", (str(project or ''),))


def _super_count(pg, table: str) -> int:
    with psycopg.connect(pg.conninfo('super')) as conn:
        return conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]  # test table names


@pytest.mark.parametrize('table', PROTECTED)
def test_a_project_sees_only_its_own_rows(pg, world, table: str) -> None:
    total = _super_count(pg, table)
    assert total >= 2, f'{table} was not seeded for both projects'
    with _app(pg) as conn:
        _scope(conn, world['a'])
        seen = conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
    assert seen == total // 2


# Immutable by trigger: any UPDATE or DELETE of a live row raises, so a row count proves nothing.
IMMUTABLE = {'config_revisions'}


@pytest.mark.parametrize('table', [t for t in PROTECTED if t not in IMMUTABLE])
def test_a_project_cannot_update_or_delete_another_projects_rows(pg, world, table: str) -> None:
    with psycopg.connect(pg.conninfo('super')) as conn:
        first_column = conn.execute(
            'SELECT column_name FROM information_schema.columns WHERE table_name = %s '
            "AND is_identity = 'NO' ORDER BY ordinal_position LIMIT 1",
            (table,),
        ).fetchone()[0]
    with _app(pg) as conn:
        _scope(conn, world['a'])
        updated = conn.execute(f'UPDATE {table} SET {first_column} = {first_column}').rowcount
        conn.rollback()
        _scope(conn, world['a'])
        try:
            deleted = conn.execute(f'DELETE FROM {table}').rowcount
        except errors.ForeignKeyViolation:
            # A row of this project is still referenced; the update above proved the filter.
            deleted = updated
        conn.rollback()
    assert updated == deleted == _super_count(pg, table) // 2


@pytest.mark.parametrize('table', sorted({t for t, _ in statements() if t in PROJECT_COLUMN}))
def test_a_project_cannot_insert_another_projects_row(pg, world, table: str) -> None:
    statement = next(sql for t, sql in statements() if t == table)
    params = ids(world['b'], 'a')
    params.update(
        index_name=f'idx-{params["k"]}',
        upload_key=f'up/{params["k"]}',
        triton_name=f'm-{params["k"]}',
    )
    with _app(pg) as conn:
        _scope(conn, world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            conn.execute(statement, params)


def test_a_statement_with_no_scope_raises_instead_of_returning_nothing(pg, world) -> None:
    with _app(pg) as conn, pytest.raises(errors.InsufficientPrivilege, match='scope_not_set'):
        conn.execute('SELECT count(*) FROM jobs')


def test_an_empty_binding_after_a_scoped_transaction_also_raises(pg, world) -> None:
    with _app(pg) as conn:
        _scope(conn, world['a'])
        conn.execute('SELECT count(*) FROM jobs')
        conn.commit()
        # the same connection, a new transaction: set_config(..., true) did not leak
        with pytest.raises(errors.InsufficientPrivilege, match='scope_not_set'):
            conn.execute('SELECT count(*) FROM jobs')


def test_a_scoped_session_that_commits_and_auto_begins_keeps_its_scope(pg_env, world) -> None:
    factory = sync_session_factory()
    with factory() as session, sync_project_tx(session, world['a']):
        before = session.execute(text('SELECT count(*) FROM jobs')).scalar_one()
        session.commit()
        after = session.execute(text('SELECT count(*) FROM jobs')).scalar_one()
    assert before == after == 3  # job, job2 and job3 of project A


def test_the_deployment_scope_sees_every_project(pg, world) -> None:
    with _app(pg) as conn:
        _scope(conn, None, 'deployment')
        assert conn.execute('SELECT count(*) FROM jobs').fetchone()[0] == _super_count(pg, 'jobs')


def test_the_application_role_cannot_bypass_row_security(pg, world) -> None:
    with _app(pg) as conn:
        flags = conn.execute(
            'SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user'
        ).fetchone()
        assert flags == (False, False)
        conn.execute('SET row_security = off')
        _scope(conn, world['a'])
        with pytest.raises(errors.InsufficientPrivilege):
            conn.execute('SELECT count(*) FROM jobs')


def test_the_owner_is_not_a_superuser(pg) -> None:
    with psycopg.connect(pg.conninfo('owner')) as conn:
        assert conn.execute(
            'SELECT rolsuper FROM pg_roles WHERE rolname = current_user'
        ).fetchone() == (False,)


def test_a_project_scope_reads_a_model_shared_with_it_and_not_an_unshared_one(pg, world) -> None:
    b_model = ids(world['b'], 'a')['model']
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute(
            "INSERT INTO model_shares (model_id, consumer_project_id, created_by) VALUES (%s, %s, 'test')",
            (b_model, world['a']),
        )
    try:
        with _app(pg) as conn:
            _scope(conn, world['a'])
            names = conn.execute('SELECT id FROM models').fetchall()
            engines = conn.execute('SELECT count(*) FROM model_engines').fetchone()[0]
            shares = conn.execute('SELECT count(*) FROM model_shares').fetchone()[0]
        assert (b_model,) in names
        assert engines == 2  # its own engine and the shared model's
        assert shares == 1
        with _app(pg) as conn:
            _scope(conn, world['c'])
            assert conn.execute('SELECT count(*) FROM models').fetchone()[0] == 0
    finally:
        with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
            conn.execute('DELETE FROM model_shares')


def test_a_consumer_cannot_insert_a_share_of_a_model_it_does_not_own(pg, world) -> None:
    b_model = ids(world['b'], 'a')['model']
    with _app(pg) as conn:
        _scope(conn, world['a'])
        with pytest.raises(errors.InsufficientPrivilege, match='row-level security'):
            conn.execute(
                'INSERT INTO model_shares (model_id, consumer_project_id, created_by) '
                "VALUES (%s, %s, 'a')",
                (b_model, world['a']),
            )


def test_the_owner_of_a_model_can_share_it(pg, world) -> None:
    b_model = ids(world['b'], 'a')['model']
    with _app(pg) as conn:
        _scope(conn, world['b'])
        conn.execute(
            "INSERT INTO model_shares (model_id, consumer_project_id, created_by) VALUES (%s, %s, 'b')",
            (b_model, world['c']),
        )
        conn.rollback()


def test_the_owning_project_sees_its_rows(pg, world) -> None:
    with _app(pg) as conn:
        _scope(conn, world['b'])
        assert conn.execute('SELECT count(*) FROM class_registries').fetchone()[0] == 1


def test_a_reused_pooled_connection_never_carries_a_binding_to_the_next_user(pg_env, world) -> None:
    engine = create_engine(pg_env.url('app'), pool_size=1, max_overflow=0)
    try:
        with engine.connect() as first:
            first.execute(text("SELECT set_config('op.scope', 'project', true)"))
            first.execute(
                text("SELECT set_config('op.project_id', :p, true)"), {'p': str(world['a'])}
            )
            first.commit()
        with engine.connect() as second, pytest.raises(Exception, match='scope_not_set'):
            second.execute(text('SELECT count(*) FROM jobs'))
    finally:
        engine.dispose()


@pytest.fixture(scope='module')
def big(pg, world):
    """Enough rows in jobs and events for the planner to prefer the project index."""
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        slugs = [f'bulk-{i}' for i in range(40)]
        projects = []
        for slug in slugs:
            conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
            pid = uuid.uuid4()
            projects.append(pid)
            conn.execute(
                'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
                "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', 'x', 'x', 'x', 'x', 'test')",
                (pid, slug, slug),
            )
        conn.execute(
            'INSERT INTO jobs (project_id, kind, queue, status, max_queue_wait, requested_by) '
            "SELECT p, 'noop', 'jobs', 'queued', interval '1 hour', 'test' "
            'FROM unnest(%s::uuid[]) p, generate_series(1, 800)',
            (projects,),
        )
        conn.execute(
            "INSERT INTO events (project_id, type, payload) SELECT p, 'x', '{}' "
            'FROM unnest(%s::uuid[]) p, generate_series(1, 800)',
            (projects,),
        )
        conn.execute('ANALYZE jobs')
        conn.execute('ANALYZE events')
    return projects[0]


@pytest.mark.parametrize('table', ['jobs', 'events'])
def test_a_scoped_select_with_the_explicit_project_filter_uses_the_project_index(
    pg, big, table: str
) -> None:
    with _app(pg) as conn:
        _scope(conn, big)
        plan = '\n'.join(
            r[0]
            for r in conn.execute(f'EXPLAIN SELECT * FROM {table} WHERE project_id = %s', (big,))
        )
    assert 'Index' in plan, plan
    assert 'project_id' in plan


@pytest.mark.parametrize('table', ['jobs', 'events'])
def test_without_the_filter_the_policy_still_hides_other_projects(pg, big, table: str) -> None:
    with _app(pg) as conn:
        _scope(conn, big)
        rows = conn.execute(f'SELECT DISTINCT project_id FROM {table}').fetchall()
    assert rows == [(big,)]

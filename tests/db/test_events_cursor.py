"""The (xid, id) cursor never skips a row that commits late, and both streams use their index."""

from __future__ import annotations

import asyncio
import uuid

import psycopg
import pytest

from src.db.repos import events as repo
from src.db.session import async_session_factory, project_tx


pytestmark = pytest.mark.db


@pytest.fixture(scope='module')
def pg(pg_events):
    """The cursor waits for the cluster's oldest open transaction: use a server of its own."""
    return pg_events


def _new_project(pg) -> uuid.UUID:
    project, slug = uuid.uuid4(), f'cur-{uuid.uuid4().hex[:8]}'
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
        conn.execute(
            'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
            "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
            (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
        )
    return project


def _insert(conn, project, kind: str) -> int:
    return conn.execute(
        'INSERT INTO events (project_id, type, payload) VALUES (%s, %s, %s::jsonb) RETURNING id',
        (project, kind, '{}'),
    ).fetchone()[0]


async def _poll(project, cursor):
    async with async_session_factory()() as session, project_tx(session, project):
        return await repo.fetch_after(session, project, *cursor)


async def _poll_for(project, cursor, count: int):
    """Poll until ``count`` rows are visible. The xmin horizon is cluster-wide, so a long
    transaction in another test's database (parallel workers) delays visibility."""
    async with asyncio.timeout(30):
        while len(rows := await _poll(project, cursor)) < count:
            await asyncio.sleep(0.2)
    return rows


@pytest.mark.asyncio
async def test_two_writers_committing_out_of_id_order_are_each_emitted_once_in_cursor_order(
    pg_env,
) -> None:
    project = _new_project(pg_env)
    first = psycopg.connect(pg_env.conninfo('super'))
    second = psycopg.connect(pg_env.conninfo('super'))
    try:
        id_first = _insert(first, project, 'a')  # takes the lower id and the older transaction
        id_second = _insert(second, project, 'b')
        second.commit()  # commits first, with the higher id
        assert id_first < id_second

        # The reader polls between the commits: the older transaction still runs, so nothing
        # may be emitted yet; emitting b now would let a commit behind the cursor and be lost.
        assert await _poll(project, (0, 0)) == []

        first.commit()
        rows = await _poll_for(project, (0, 0), 2)
    finally:
        first.close()
        second.close()
    assert [r.type for r in rows] == ['a', 'b']
    assert [r.id for r in rows] == [id_first, id_second]

    last = rows[-1]
    assert await _poll(project, (last.xid, last.id)) == []


@pytest.mark.asyncio
async def test_a_single_writer_is_emitted_in_id_order(pg_env) -> None:
    project = _new_project(pg_env)
    with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
        ids = [_insert(conn, project, f'e{i}') for i in range(5)]
    rows = await _poll_for(project, (0, 0), 5)
    assert [r.id for r in rows] == ids


@pytest.mark.asyncio
async def test_paging_resumes_exactly_after_the_cursor(pg_env) -> None:
    project = _new_project(pg_env)
    with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
        for i in range(7):
            _insert(conn, project, f'e{i}')
    await _poll_for(project, (0, 0), 7)
    async with async_session_factory()() as session, project_tx(session, project):
        page = await repo.fetch_after(session, project, 0, 0, limit=3)
        rest = await repo.fetch_after(session, project, page[-1].xid, page[-1].id, limit=10)
        head = await repo.head_cursor(session, project)
    assert [r.type for r in page + rest] == [f'e{i}' for i in range(7)]
    assert head == (rest[-1].xid, rest[-1].id)


def test_both_cursor_queries_use_their_index(pg_env) -> None:
    project = _new_project(pg_env)
    with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
        for _ in range(3):
            _insert(conn, project, 'x')
            _insert(conn, None, 'g')
        conn.execute('ANALYZE events')
        conn.execute('SET enable_seqscan = off')
        scoped = repo._page_sql(project).replace(':project_id', '%(project_id)s')
        global_ = repo._page_sql(None)
        for sql, params in (
            (scoped, {'project_id': project}),
            (global_, {}),
        ):
            literal = sql.replace(':xid', "'0'").replace(':id', '0').replace(':limit', '500')
            plan = '\n'.join(r[0] for r in conn.execute('EXPLAIN ' + literal, params))
            assert 'Index' in plan, plan
            assert 'Seq Scan' not in plan, plan

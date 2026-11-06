"""Pending rows: the signature every writer calls, and the writers' ordering."""

from __future__ import annotations

import psycopg
import pytest

from src.db.session import async_session_factory, project_tx
from src.services.near_duplicates.pending import PendingError, enqueue_pending


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _enqueue(project, level, ids, reason):
    async with async_session_factory()() as session, project_tx(session, project):
        return await enqueue_pending(session, project, level, ids, reason)


def _rows(pg, project, level='item'):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        return c.execute(
            'SELECT member_id, reason, claimed_by_job_id FROM near_dup_pending '
            'WHERE project_id = %s AND level = %s ORDER BY member_id',
            (project, level),
        ).fetchall()


async def test_one_row_per_member_and_the_newest_reason_wins(pg, project) -> None:
    assert await _enqueue(project, 'item', ['b', 'a', 'a'], 'added') == 2
    await _enqueue(project, 'item', ['a'], 'vector_changed')
    await _enqueue(project, 'image', ['a'], 'deleted')
    assert _rows(pg, project) == [('a', 'vector_changed', None), ('b', 'added', None)]
    assert _rows(pg, project, 'image') == [('a', 'deleted', None)]


async def test_a_change_during_a_rescore_releases_the_claim(pg, project) -> None:
    await _enqueue(project, 'item', ['a'], 'added')
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        job = c.execute(
            'INSERT INTO jobs (project_id, kind, queue, status, max_queue_wait, requested_by) '
            "VALUES (%s, 'noop', 'jobs', 'queued', interval '1 hour', 'test') RETURNING id",
            (project,),
        ).fetchone()[0]
        c.execute(
            'UPDATE near_dup_pending SET claimed_by_job_id = %s WHERE project_id = %s',
            (job, project),
        )
    await _enqueue(project, 'item', ['a'], 'boxes_changed')
    assert _rows(pg, project) == [('a', 'boxes_changed', None)]


async def test_thousands_of_members_in_one_call(pg, project) -> None:
    ids = [f'm{i:05d}' for i in range(4500)]
    assert await _enqueue(project, 'image', ids, 'added') == 4500
    assert len(_rows(pg, project, 'image')) == 4500


async def test_core_members_have_no_project(project, pg) -> None:
    from src.db.session import core_tx

    async with async_session_factory()() as session, core_tx(session, 'test core pending'):
        await enqueue_pending(session, None, 'core', ['x'], 'added')
        await enqueue_pending(session, None, 'core', ['x'], 'deleted')
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        assert c.execute(
            "SELECT reason FROM near_dup_pending WHERE project_id IS NULL AND member_id = 'x'"
        ).fetchall() == [('deleted',)]


async def test_an_unknown_level_or_reason_is_refused(project) -> None:
    with pytest.raises(PendingError):
        await _enqueue(project, 'frame', ['a'], 'added')
    with pytest.raises(PendingError):
        await _enqueue(project, 'item', ['a'], 'moved')

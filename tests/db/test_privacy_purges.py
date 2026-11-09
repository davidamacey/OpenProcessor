"""Revision 0004_privacy_purges: the counts-only purge record and its one-time jsonl import."""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING

import psycopg
import pytest
from psycopg import errors

from src.db.repos import privacy_purges
from src.services.privacy_audit import record_purge
from src.services.upgrade.privacy_audit_import import import_audit_files


if TYPE_CHECKING:
    from pathlib import Path

    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        c.execute('DELETE FROM privacy_purges')
        yield c


def test_the_table_refuses_counts_that_are_not_numbers_or_flags(conn) -> None:
    ok = "INSERT INTO privacy_purges (kind, counts, actor, host) VALUES (%s, %s, 'a', 'h')"
    conn.execute(ok, ('faces', json.dumps({'face_docs': 3, 'index_deleted': True})))
    with pytest.raises(errors.CheckViolation):
        conn.execute(ok, ('faces', json.dumps({'name': 'Alice'})))
    with pytest.raises(errors.CheckViolation):
        conn.execute(ok, ('faces', json.dumps([1])))
    with pytest.raises(errors.CheckViolation):
        conn.execute(ok, ('other', '{}'))


@pytest.mark.asyncio
async def test_record_purge_writes_one_row(pg_env, bound_async_engine, conn) -> None:
    await record_purge('ocr', {'lines': 4})
    rows = conn.execute('SELECT kind, counts, actor, host FROM privacy_purges').fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 'ocr'
    assert rows[0][1] == {'lines': 4}
    assert rows[0][2]
    assert rows[0][3]


def test_the_repo_refuses_a_non_numeric_count() -> None:
    with pytest.raises(ValueError, match='counts'):
        privacy_purges.check_counts({'face_docs': 'Alice'})


@pytest.mark.asyncio
async def test_the_import_is_idempotent_and_leaves_the_files(
    pg_env, bound_async_engine, conn, tmp_path: Path
) -> None:
    from src.db.session import async_session_factory

    audit = tmp_path / 'data' / 'audit'
    audit.mkdir(parents=True)
    faces = audit / 'faces_purge.jsonl'
    faces.write_text(
        json.dumps(
            {
                'face_docs': 5,
                'index_deleted': True,
                'at': '2026-11-01T10:00:00+00:00',
                'version': '0.7.0',
            }
        )
        + '\n'
    )
    ocr = audit / 'ocr_purge.jsonl'
    ocr.write_text(json.dumps({'lines': 2, 'at': '2026-11-02T10:00:00+00:00'}) + '\n')
    before = (faces.read_text(), ocr.read_text())
    for expected in (2, 0):
        async with async_session_factory()() as session, session.begin():
            assert await import_audit_files(session, audit) == expected
    rows = conn.execute('SELECT kind, counts, at FROM privacy_purges ORDER BY kind').fetchall()
    assert [r[0] for r in rows] == ['faces', 'ocr']
    assert rows[0][1] == {'face_docs': 5, 'index_deleted': True}
    assert rows[1][2] == dt.datetime(2026, 11, 2, 10, tzinfo=dt.UTC)
    assert (faces.read_text(), ocr.read_text()) == before


@pytest.mark.asyncio
async def test_a_missing_audit_directory_imports_nothing(
    pg_env, bound_async_engine, conn, tmp_path: Path
) -> None:
    from src.db.session import async_session_factory

    async with async_session_factory()() as session, session.begin():
        assert await import_audit_files(session, tmp_path / 'none') == 0

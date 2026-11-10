"""The face and OCR purge scripts against a real PostgreSQL.

Only OpenSearch is faked (the in-memory fakes of the offline purge tests). The audit row is
written by the real ``record_purge`` through the real ``run_with_engine``, and read back as the
superuser, so a purge that deleted data but wrote no row, or a wrong row, fails here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from test_face_purge import FACES, SECRETS, FakeOpenSearch as FakeFaceSearch
from test_ocr_purge import OCR, SECRET, FakeOpenSearch as FakeOcrSearch

from scripts import purge_face_data, purge_ocr_data


if TYPE_CHECKING:
    from pathlib import Path

    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        c.execute('DELETE FROM privacy_purges')
        yield c


def _rows(conn: psycopg.Connection) -> list[tuple[str, dict[str, Any]]]:
    return conn.execute('SELECT kind, counts FROM privacy_purges ORDER BY id').fetchall()


def test_ocr_purge_deletes_and_writes_the_audit_row(
    pg_env: PgDatabase, conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOcrSearch()
    monkeypatch.setattr(purge_ocr_data, 'make_client', lambda: fake)
    assert purge_ocr_data.main(['--yes']) == 0
    assert [m[:2] for m in fake.mutations] == [
        ('delete_by_query', OCR),
        ('forcemerge', OCR),
    ]
    assert fake.docs[OCR] == []
    assert _rows(conn) == [('ocr', {'lines': 4})]
    assert SECRET not in json.dumps(_rows(conn))


def test_ocr_dry_run_and_refusal_write_no_row(
    pg_env: PgDatabase, conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeOcrSearch()
    monkeypatch.setattr(purge_ocr_data, 'make_client', lambda: fake)
    assert purge_ocr_data.main(['--dry-run']) == 0
    assert purge_ocr_data.main([]) == 2
    assert fake.mutations == []
    assert _rows(conn) == []


def test_face_purge_deletes_and_writes_the_audit_row(
    pg_env: PgDatabase,
    conn: psycopg.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / 'faiss'
    index_dir.mkdir()
    (index_dir / 'faces.index').write_bytes(b'x')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    fake = FakeFaceSearch()
    monkeypatch.setattr(purge_face_data, 'make_client', lambda: fake)
    assert purge_face_data.main(['--yes', '--index-dir', str(index_dir)]) == 0
    assert [m for m in fake.mutations if m[0] == 'indices.delete'] == [('indices.delete', FACES)]
    assert not (index_dir / 'faces.index').exists()
    ((kind, counts),) = _rows(conn)
    assert kind == 'faces'
    assert counts['face_docs'] == 5
    assert counts['persons'] == 3
    written = json.dumps(counts)
    for secret in (*SECRETS, str(index_dir), 'f0'):
        assert secret not in written


def test_face_dry_run_writes_no_row(
    pg_env: PgDatabase,
    conn: psycopg.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake = FakeFaceSearch()
    monkeypatch.setattr(purge_face_data, 'make_client', lambda: fake)
    assert purge_face_data.main(['--dry-run', '--index-dir', str(tmp_path)]) == 0
    assert fake.mutations == []
    assert _rows(conn) == []

"""A 0.6.0 pending queue row arrives under the file key the 0.7.0 scanner writes, so the first
scan after the upgrade finds it already queued and does not ingest the file twice."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from src.services.curation.bulk_ingest.source_queue import QueueEntry, file_key_for
from src.services.upgrade.from_060.parse_jobs import pending_queue_rows


if TYPE_CHECKING:
    from pathlib import Path

ROWS = [
    ('photos/a b.jpg', 'etag-1', 'v1', 10, 'pending', 0, None, 1.0, 2.0),
    ('plain/c.jpg', '12345:678', None, 11, 'pending', 1, 'timeout', 3.0, 4.0),
]


def _image(tmp_path: Path) -> bytes:
    path = tmp_path / 'queue.sqlite'
    conn = sqlite3.connect(path)
    conn.execute(
        'CREATE TABLE objects (key TEXT NOT NULL, etag TEXT NOT NULL, version TEXT, size INTEGER,'
        ' state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT,'
        ' queued_at REAL NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY (key, etag))'
    )
    conn.executemany('INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?)', ROWS)
    conn.commit()
    conn.close()
    return path.read_bytes()


def _keys(tmp_path: Path, backend: str, bucket: str) -> list[str]:
    rows = pending_queue_rows(None, _image(tmp_path), backend=backend, bucket=bucket)
    return [row[1] for row in rows]


def _scanner_key(backend: str, bucket: str, row: tuple) -> str:
    entry = QueueEntry(key=row[0], etag=row[1], version=row[2], size=row[3])
    return file_key_for(entry, backend=backend, bucket=bucket)


def test_an_s3_row_becomes_the_object_reference_of_the_sources_bucket(tmp_path: Path) -> None:
    keys = _keys(tmp_path, 's3', 'shots')
    assert keys == [_scanner_key('s3', 'shots', r) for r in ROWS]
    assert keys[0].startswith('s3://shots/photos/a%20b.jpg#etag=etag-1')


def test_a_folder_row_becomes_the_path_and_its_version(tmp_path: Path) -> None:
    keys = _keys(tmp_path, 'folder', '')
    assert keys == ['photos/a b.jpg\x1fetag-1', 'plain/c.jpg\x1f12345:678']
    assert keys == [_scanner_key('folder', '', r) for r in ROWS]

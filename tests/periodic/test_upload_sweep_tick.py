"""``op.periodic.upload_sweep``: old extracted uploads go unless a live upload row or any import
still names them (by upload, or by the ``source_sha`` of a directory-source import)."""

from __future__ import annotations

import os
import time
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fixtures.job_rows import make_project, sql

from src.services.periodic import upload_sweep
from src.services.periodic.runner import run_periodic


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env')]

LONG_AGO = time.time() - 30 * 24 * 3600


def sha(letter: str) -> str:
    return letter * 64


def make_dir(root: Path, letter: str) -> Path:
    path = root / 'datasets' / sha(letter)[:16]
    path.mkdir(parents=True)
    (path / 'image.jpg').write_bytes(b'x')
    os.utime(path, (LONG_AGO, LONG_AGO))
    return path


def add_upload(pg, project: uuid.UUID, letter: str, *, expired: bool) -> uuid.UUID:
    upload = uuid.uuid4()
    when = "now() - interval '1 day'" if expired else "now() + interval '1 day'"
    sql(
        pg,
        'INSERT INTO dataset_uploads (id, project_id, filename, size_bytes, sha256, storage_key, '
        f"status, expires_at) VALUES (%s, %s, 'a.zip', 1, %s, %s, 'extracted', {when})",
        upload,
        project,
        sha(letter),
        f'k/{upload}',
    )
    return upload


def add_import(
    pg, project: uuid.UUID, *, upload: uuid.UUID | None = None, scan_sha: str = ''
) -> None:
    source_key = None if upload else f'dir/{uuid.uuid4()}'
    scan = f'{{"source_sha": "{scan_sha}"}}' if scan_sha else '{}'
    sql(
        pg,
        'INSERT INTO dataset_imports (project_id, upload_id, source_key, format, mapping, options, '
        "scan) VALUES (%s, %s, %s, 'yolo', '{}', '{}', %s::jsonb)",
        project,
        upload,
        source_key,
        scan,
    )


def test_only_unreferenced_old_uploads_are_removed(pg, tmp_path, monkeypatch) -> None:
    project = make_project(pg)
    monkeypatch.setattr(
        upload_sweep, 'get_curation_config', lambda: SimpleNamespace(upload_root=tmp_path)
    )
    live = make_dir(tmp_path, 'a')  # its row has not expired
    add_upload(pg, project, 'a', expired=False)
    imported = make_dir(tmp_path, 'b')  # expired, but an import was made from it
    add_import(pg, project, upload=add_upload(pg, project, 'b', expired=True))
    by_directory = make_dir(tmp_path, 'c')  # a directory-source import copied into it
    add_import(pg, project, scan_sha=sha('c'))
    orphan = make_dir(tmp_path, 'd')
    expired_alone = make_dir(tmp_path, 'e')
    add_upload(pg, project, 'e', expired=True)

    run_periodic('upload_sweep', upload_sweep.tick())

    assert [p.exists() for p in (live, imported, by_directory, orphan, expired_alone)] == [
        True,
        True,
        True,
        False,
        False,
    ]

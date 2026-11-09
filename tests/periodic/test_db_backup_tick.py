"""``op.periodic.db_backup``: a ``pg_dump`` of the control-plane database into
``OP_DB_BACKUP_DIR``, the newest ``OP_DB_BACKUP_KEEP`` kept, one dump at a time, and a failed or
empty dump leaves no file behind. ``pg_dump`` itself is faked; its argument list is asserted.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.config.advisory_locks import DB_BACKUP
from src.config.env import EnvError
from src.services.periodic import db_backup
from src.services.periodic.runner import run_periodic


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_env')]


class FakeDump:
    def __init__(self, content: bytes = b'PGDMP', fail: bool = False) -> None:
        self.content, self.fail, self.calls = content, fail, []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((argv, kwargs))
        if self.fail:
            raise subprocess.CalledProcessError(1, argv, stderr=b'connection refused')
        Path(argv[argv.index('--file') + 1]).write_bytes(self.content)
        return subprocess.CompletedProcess(argv, 0)


@pytest.fixture
def dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeDump:
    fake = FakeDump()
    monkeypatch.setattr(db_backup.subprocess, 'run', fake)
    monkeypatch.setenv('OP_DB_BACKUP_DIR', str(tmp_path / 'backups'))
    monkeypatch.setenv('OP_DB_BACKUP_KEEP', '2')
    return fake


def tick(hour: int) -> Path | None:
    stamp = dt.datetime(2026, 11, 8, hour, tzinfo=dt.UTC)
    return run_periodic('db_backup', db_backup.tick(now=stamp))


def names(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / 'backups').iterdir())


def test_a_tick_writes_a_custom_format_dump_as_the_owner_and_keeps_the_newest(
    dump: FakeDump, tmp_path: Path
) -> None:
    for hour in (1, 2, 3):
        tick(hour)
    assert names(tmp_path) == [
        'openprocessor-20261108T020000Z.dump',
        'openprocessor-20261108T030000Z.dump',
    ]
    argv, kwargs = dump.calls[0]
    assert argv[0] == 'pg_dump'
    assert '--format=custom' in argv
    assert argv[argv.index('--username') + 1] == 'op_owner'
    assert 'PGPASSWORD' in kwargs['env']
    assert not any('PGPASSWORD' in a for a in argv)


def test_a_failed_dump_leaves_no_file_and_fails_the_tick(dump: FakeDump, tmp_path: Path) -> None:
    dump.fail = True
    with pytest.raises(db_backup.BackupError, match='connection refused'):
        tick(1)
    assert names(tmp_path) == []


def test_an_empty_dump_is_refused(dump: FakeDump, tmp_path: Path) -> None:
    dump.content = b''
    with pytest.raises(db_backup.BackupError, match='empty'):
        tick(1)
    assert names(tmp_path) == []


def test_the_backup_directory_is_required(dump: FakeDump, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_DB_BACKUP_DIR')
    with pytest.raises(EnvError):
        tick(1)


def test_a_tick_that_meets_the_lock_held_by_another_takes_no_dump(
    dump: FakeDump, pg, tmp_path: Path
) -> None:
    import psycopg

    with psycopg.connect(pg.conninfo('super'), autocommit=True) as holder:
        holder.execute('SELECT pg_advisory_lock(%s, 0)', (DB_BACKUP,))
        assert tick(1) is None
    assert dump.calls == []
    assert not (tmp_path / 'backups').exists() or names(tmp_path) == []

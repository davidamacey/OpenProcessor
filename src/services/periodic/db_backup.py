"""``op.periodic.db_backup``: a ``pg_dump`` of the control-plane database, when asked for.

Scheduled only when ``OP_DB_BACKUP_INTERVAL_H`` is set (off by default, D-61). The tick takes the
``DB_BACKUP`` advisory lock and runs ``pg_dump -Fc`` (compressed, restorable selectively,
consistent without blocking writers: https://www.postgresql.org/docs/current/app-pgdump.html) as
the owner role into ``OP_DB_BACKUP_DIR`` (required: the directory is a mount that does not share
the Postgres volume), under a ``.part`` name renamed on success, then keeps the newest
``OP_DB_BACKUP_KEEP`` dumps. The lock's transaction is kept busy with ``SELECT 1`` while the dump
runs so the server's idle-in-transaction timeout cannot end it and let a second tick start a
second dump. The password reaches ``pg_dump`` only through its environment.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import subprocess
from typing import TYPE_CHECKING

from sqlalchemy import text

from src.config.advisory_locks import DB_BACKUP
from src.config.database import DatabaseSettings
from src.config.env import EnvError, env_int, env_optional_path, env_str
from src.core.logging import get_logger
from src.db.engine import current_async_engine


if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

KEEP_ENV = 'OP_DB_BACKUP_KEEP'
DIR_ENV = 'OP_DB_BACKUP_DIR'
DEFAULT_KEEP = 7
KEEPALIVE_S = 30.0
DUMP_TIMEOUT_S = 3000  # under the task's soft limit, so a stuck dump ends as BackupError
PREFIX = 'openprocessor-'
SUFFIX = '.dump'
_LOCK = text('SELECT pg_try_advisory_xact_lock(:class, 0)')


class BackupError(RuntimeError):
    """``pg_dump`` failed or the backup directory is unusable; nothing was kept."""


def backup_dir() -> Path:
    directory = env_optional_path(DIR_ENV)
    if directory is None:
        raise EnvError(DIR_ENV, '', 'is required for the periodic database backup')
    return directory


def _command(settings: DatabaseSettings, target: Path) -> list[str]:
    return [
        'pg_dump',
        '--host',
        settings.direct_host,
        '--port',
        str(settings.port),
        '--username',
        settings.user('owner'),
        '--dbname',
        settings.name,
        '--format=custom',
        '--file',
        str(target),
    ]


def dump_and_prune(settings: DatabaseSettings, directory: Path, keep: int, stamp: str) -> Path:
    """Write one dump, then delete all but the newest ``keep``; the final path."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = directory / f'{PREFIX}{stamp}{SUFFIX}'
    part = final.with_name(final.name + '.part')
    env = {
        'PATH': env_str('PATH'),
        'PGPASSWORD': settings.password('owner'),
        'PGSSLMODE': settings.sslmode,
    }
    try:
        subprocess.run(  # nosec B603 - fixed argument list, no shell
            _command(settings, part),
            env=env,
            check=True,
            capture_output=True,
            timeout=DUMP_TIMEOUT_S,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        part.unlink(missing_ok=True)
        stderr = getattr(exc, 'stderr', None)
        detail = stderr.decode(errors='replace').strip() if isinstance(stderr, bytes) else ''
        raise BackupError(f'pg_dump failed: {exc} {detail}'.strip()) from exc
    if part.stat().st_size == 0:
        part.unlink()
        raise BackupError('pg_dump wrote an empty file')
    part.replace(final)
    for old in sorted(directory.glob(f'{PREFIX}*{SUFFIX}'), reverse=True)[keep:]:
        old.unlink(missing_ok=True)
    return final


async def tick(*, now: dt.datetime | None = None) -> Path | None:
    """The new dump's path, or ``None`` when another tick holds the lock."""
    directory = backup_dir()
    keep = env_int(KEEP_ENV, DEFAULT_KEEP, minimum=1)
    stamp = (now or dt.datetime.now(dt.UTC)).strftime('%Y%m%dT%H%M%SZ')
    settings = DatabaseSettings.from_env()
    async with current_async_engine().connect() as conn, conn.begin():
        if not (await conn.execute(_LOCK, {'class': DB_BACKUP})).scalar_one():
            return None
        dump = asyncio.ensure_future(
            asyncio.to_thread(dump_and_prune, settings, directory, keep, stamp)
        )
        try:
            while True:
                done, _ = await asyncio.wait({dump}, timeout=KEEPALIVE_S)
                if done:
                    path = dump.result()
                    logger.info('db_backup_written', path=str(path))
                    return path
                await conn.execute(text('SELECT 1'))
        finally:
            if not dump.done():
                await asyncio.wait({dump})

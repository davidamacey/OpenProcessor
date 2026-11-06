#!/usr/bin/env python3
"""The one-step 0.6.0 -> 0.7.0 upgrade of the control data (called by ``openprocessor upgrade``).

    python -m scripts.db.upgrade_from_060 [--reset] [--install-dir DIR]

Runs ``export_bundle``, ``load``, ``verify`` and ``delete_sources`` in order and resumes at the
first step that did not complete. The 0.6.0 API and workers must be stopped, OpenSearch up and
PostgreSQL at head. A verify that fails exits 1 with the report path and deletes nothing; fix the
cause and run again, or run with ``--reset`` to load again from the same bundle.

Exit codes: 0 done (or already done), 1 verify failed, 2 anything else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.clients.storage.roots import build_rooted_store
from src.config.database import DatabaseSettings
from src.config.env import env_path
from src.config.storage import storage_root_paths
from src.db.engine import database_url, one_off_connection
from src.services.upgrade import gate
from src.services.upgrade.from_060.admin_client import ControlReader, ControlWriter, build_client
from src.services.upgrade.from_060.layout import Layout
from src.services.upgrade.from_060.model import UnresolvedBuiltinError
from src.services.upgrade.from_060.reset import reset
from src.services.upgrade.from_060.runner import (
    AlreadyCompletedError,
    NothingToUpgradeError,
    UpgradeResult,
    UpgradeRunner,
    VerifyFailedError,
)


DEFAULT_INSTALL_DIR = '.install'


def summary_lines(result: UpgradeResult) -> str:
    """What the installer prints after a finished upgrade."""
    counts = result.load_counts
    drift = result.verify.class_drift if result.verify else {}
    off = sum(d['unknown_class_id'] + d['name_disagrees'] for d in drift.values())
    failed = result.verify.failed_projects if result.verify else []
    lines = [
        (
            f'projects moved: {counts.get("projects", 0)}  classes: {counts.get("project_classes", 0)}  '
            f'configs: {counts.get("config_documents", 0)}'
        ),
        f'jobs recorded as interrupted: {counts.get("interrupted_jobs", 0)}',
        f'items whose class disagrees with the registry: {off} (reported, not changed)',
        f'projects whose indexes are missing (status failed): {", ".join(failed) or "none"}',
        f'bundle: {result.bundle}',
        f'report: {result.report_path}',
    ]
    return '\n'.join(lines) + '\n'


async def run(install_dir: Path) -> int:
    # The schema owner, as the migration runner is: COPY FROM is refused on a table with row-level
    # security for a role the policies apply to, and the upgrade is an administrative one-shot.
    engine = create_async_engine(
        database_url(
            DatabaseSettings.from_env(), 'owner', direct=True, application_name='op-upgrade-0.6.0'
        ),
        poolclass=pool.NullPool,
    )
    client = build_client()
    try:
        runner = UpgradeRunner(
            sessions=async_sessionmaker(engine, expire_on_commit=False),
            reader=ControlReader(client),
            writer=ControlWriter(client),
            layout=Layout.from_env(),
            store=build_rooted_store(storage_root_paths()),
            install_dir=install_dir,
        )
        try:
            result = await runner.run()
        except AlreadyCompletedError:
            sys.stdout.write('the 0.6.0 upgrade already completed\n')
            return 0
        except NothingToUpgradeError as exc:
            sys.stdout.write(f'{exc}\n')
            return 0
        except UnresolvedBuiltinError as exc:
            sys.stderr.write(f'the upgrade stopped, nothing was changed: {exc}\n')
            return 2
        except VerifyFailedError as exc:
            sys.stderr.write(f'{exc}\n')
            sys.stdout.write(
                json.dumps({'status': 'verify_failed', 'report': str(exc.report_path)}) + '\n'
            )
            return 1
        sys.stderr.write(summary_lines(result))
        summary = {
            'status': 'completed',
            'bundle': str(result.bundle),
            'report': str(result.report_path),
            'counts': result.load_counts,
            'steps': result.steps_run,
            'class_drift': (result.verify.class_drift if result.verify else {}),
            'failed_projects': (result.verify.failed_projects if result.verify else []),
        }
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        return 0
    finally:
        await client.close()
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--reset', action='store_true', help='undo a committed load whose verify failed, then run'
    )
    parser.add_argument(
        '--install-dir',
        type=Path,
        default=env_path('OP_INSTALL_DIR', DEFAULT_INSTALL_DIR),
        help='where the bundle directory is written',
    )
    args = parser.parse_args(argv)
    try:
        if args.reset:
            with one_off_connection('owner', 'op-upgrade-reset') as conn:
                reset(conn)
        return asyncio.run(run(args.install_dir))
    except gate.UpgradeStateError as exc:
        sys.stderr.write(f'{exc}\n')
        return 2


if __name__ == '__main__':
    sys.exit(main())

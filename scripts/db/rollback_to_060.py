#!/usr/bin/env python3
"""Rebuild the 0.6.0 control state from the upgrade bundle (called by ``openprocessor upgrade --rollback``).

    python -m scripts.db.rollback_to_060 --bundle .install/upgrade-0.6.0-<utc>

OpenSearch must be up and every 0.6.0 and 0.7.0 API and worker stopped. Prints a JSON report;
exit 0 when everything restored matches the bundle, 1 when it does not, 2 on an error. The
installer marks the PostgreSQL data abandoned and starts the 0.6.0 services only after exit 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.clients.storage.roots import build_rooted_store
from src.config.storage import storage_root_paths
from src.services.upgrade.from_060.admin_client import ControlReader, ControlWriter, build_client
from src.services.upgrade.from_060.bundle import BundleError
from src.services.upgrade.from_060.layout import Layout
from src.services.upgrade.from_060.rollback import RollbackError, rollback


async def run(bundle: Path) -> int:
    client = build_client()
    try:
        report = await rollback(
            bundle,
            ControlReader(client),
            ControlWriter(client),
            Layout.from_env(),
            build_rooted_store(storage_root_paths()),
        )
    finally:
        await client.close()
    sys.stdout.write(
        json.dumps(
            {
                'status': 'restored' if report.ok else 'mismatch',
                'indexes': report.indexes,
                'files_restored': report.files_restored,
                'files_kept': report.files_kept,
                'registries_renamed_back': report.registries_renamed_back,
                'problems': report.problems,
            },
            indent=2,
            sort_keys=True,
        )
        + '\n'
    )
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--bundle', type=Path, required=True, help='the upgrade-0.6.0-<utc> directory'
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args.bundle))
    except (BundleError, RollbackError) as exc:
        sys.stderr.write(f'{exc}\n')
        return 2


if __name__ == '__main__':
    sys.exit(main())

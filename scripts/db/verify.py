#!/usr/bin/env python3
"""``openprocessor db verify``: the database against OpenSearch and storage (report only).

Prints a JSON report; exit 0 when it ran (drift is reported, not repaired), 2 when it could not.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from src.clients.storage.roots import build_rooted_store
from src.config.storage import storage_root_paths
from src.db.engine import run_with_engine
from src.db.session import async_session_factory, deployment_tx
from src.services.upgrade.db_verify import verify_database
from src.services.upgrade.from_060.admin_client import ControlReader, build_client


async def run() -> int:
    client = build_client()
    try:
        async with async_session_factory()() as session, deployment_tx(session, 'db verify'):
            report = await verify_database(
                session, ControlReader(client), build_rooted_store(storage_root_paths())
            )
    finally:
        await client.close()
    sys.stdout.write(json.dumps(report.to_json(), indent=2, sort_keys=True) + '\n')
    return 0


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    return asyncio.run(run_with_engine(run(), 'op-verify', one_shot=True))


if __name__ == '__main__':
    sys.exit(main())

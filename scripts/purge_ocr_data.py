#!/usr/bin/env python3
"""Report or remove every line in the core text index.

Deletes all documents of ``visual_search_ocr`` (the recognised text of core ``/ocr`` and
``/ingest`` calls, deployment-wide and outside project isolation) and merges away the deleted
segments. Curation projects keep their text in their own indexes; project delete removes it.
There is no API route, because the API has no authentication.

    python -m scripts.purge_ocr_data --dry-run   # count, changes nothing
    python -m scripts.purge_ocr_data             # count, then refuses (nothing deleted)
    python -m scripts.purge_ocr_data --yes       # delete

Exit codes: 0 done (or dry run), 1 lines remained, 2 refused.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402

from sqlalchemy.exc import SQLAlchemyError

from src.clients.opensearch.names import IndexName
from src.config import get_settings
from src.core.logging import get_logger
from src.db.engine import run_with_engine
from src.services.privacy_audit import record_purge
from src.services.projects.guard import make_script_opensearch


logger = get_logger(__name__)

_ALL: dict[str, Any] = {'query': {'match_all': {}}}


def make_client() -> Any:
    """The guarded script client, unbound: it reaches only indexes no project owns."""
    return make_script_opensearch([get_settings().opensearch_url])


async def _lines(client: Any) -> int:
    index = IndexName.OCR.value
    if not await client.indices.exists(index=index):
        return 0
    return int((await client.count(index=index, body=_ALL))['count'])


async def _audit(lines: int) -> None:
    """One counts-only record, to the log and to ``privacy_purges``."""
    logger.warning('ocr_index_purged', lines=lines)
    await record_purge('ocr', {'lines': lines})


async def _run(args: argparse.Namespace) -> int:
    client = make_client()
    try:
        before = await _lines(client)
        if args.dry_run or not args.yes:
            print(json.dumps({'lines': before}))
            if args.dry_run:
                return 0
            print('refusing: re-run with --yes to delete this', file=sys.stderr)
            return 2
        index = IndexName.OCR.value
        if before:
            await client.delete_by_query(
                index=index, body=_ALL, refresh=True, conflicts='proceed', wait_for_completion=True
            )
            await client.indices.forcemerge(index=index, only_expunge_deletes=True)
        after = await _lines(client)
        try:
            await _audit(before)
        except SQLAlchemyError as exc:
            print(json.dumps({'before': before, 'after': after}))
            print(
                f'the data is deleted but the audit record failed: {type(exc).__name__}',
                file=sys.stderr,
            )
            return 1
        print(json.dumps({'before': before, 'after': after}))
        return 0 if after == 0 else 1
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='print the count; change nothing')
    mode.add_argument('--yes', action='store_true', help='delete every line')
    return asyncio.run(
        run_with_engine(_run(parser.parse_args(argv)), 'op-purge-ocr', one_shot=True)
    )


if __name__ == '__main__':
    sys.exit(main())

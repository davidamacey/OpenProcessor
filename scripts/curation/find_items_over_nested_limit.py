#!/usr/bin/env python3
"""List the items of a project that hold too many nested documents (read-only).

OpenSearch k-NN issue 3623: a parent document with 150 or more nested documents (boxes plus their
vectors plus assist attributes) is read back with another document's box vectors, and any write
on it persists them. The writers refuse every write on such an item except one repair write, and
the cluster projection and the embedding-state backfill leave it as it is. This script finds
them; it never writes.

Exit code 0: none. Exit code 1: at least one (the list is on stdout). Which boxes to remove is a
human decision; ``--repair-plan`` prints, per item, how many boxes at least. The repair write is a
box-list edit that brings the item under the limit (``PUT .../crops/{crop_id}/regions``): it also
empties the item's box vectors, which ``backfill_region_embeddings.py`` re-creates.

    python3 scripts/curation/find_items_over_nested_limit.py --project my-project
    python3 scripts/curation/find_items_over_nested_limit.py --project my-project --repair-plan
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
import asyncio
import os
from typing import Any

from src.config import get_curation_config
from src.config.nested_limit import NESTED_DOCS_LIMIT
from src.db.engine import run_with_engine
from src.services.curation.nested_limit_scan import OverLimitItem, find_over_limit_items
from src.services.embedding_spaces.readers import vector_source_excludes
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import add_project_argument, bind_script_project


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')


REPAIR_NOTE = (
    'These items must be repaired before their box vectors are trusted: OpenSearch returns '
    "another document's vectors for them (k-NN issue 3623), and the cluster projection and the "
    'embedding-state backfill leave them as they are. The repair is one box-list write that '
    'brings the item under the limit (PUT /curation/projects/{project}/crops/{crop_id}/regions '
    'with the boxes to keep; which to remove is a human decision). It also empties the '
    "item's box vectors; backfill_region_embeddings.py --apply re-creates them. Every other "
    'write on such an item is refused.'
)


def format_item(item: OverLimitItem, *, repair_plan: bool) -> str:
    line = f'{item.crop_id}  boxes={item.boxes}  nested_docs={item.nested_docs}'
    if repair_plan:
        line += f'  remove_at_least={item.remove_at_least}'
    return line


async def run(args: argparse.Namespace, client: Any) -> int:
    items = await find_over_limit_items(
        client,
        index=args.index,
        vector_excludes=await vector_source_excludes(),
        page_size=args.page_size,
    )
    for item in items:
        print(format_item(item, repair_plan=args.repair_plan))
    print(f'{len(items)} item(s) at or over {NESTED_DOCS_LIMIT} nested documents in {args.index}')
    if items:
        print(REPAIR_NOTE)
    return 1 if items else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--index', default=None, help='Default: the bound project items index.')
    p.add_argument('--page-size', type=int, default=500)
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument(
        '--repair-plan',
        action='store_true',
        help='Also print how many boxes each item must lose (which ones is a human decision).',
    )
    add_project_argument(p)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.page_size <= 0:
        parser.error('--page-size must be positive')
    bind_script_project(args.project)
    args.index = args.index or get_curation_config().items_index

    async def _async_main() -> int:
        client = make_script_opensearch([args.opensearch_url], use_ssl=False, timeout=300)
        try:
            return await run(args, client)
        finally:
            await client.close()

    return asyncio.run(run_with_engine(_async_main(), 'op-find-items-over-nested-limit'))


if __name__ == '__main__':
    sys.exit(main())

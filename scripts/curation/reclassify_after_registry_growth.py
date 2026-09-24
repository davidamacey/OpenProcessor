#!/usr/bin/env python3
"""Reclassify unmatched VLM labels after the class registry grows.

The recurring registry-growth loop:

    1. ``GET /curation/review/unmatched_terms`` — see which raw VLM labels
       the registry could not resolve, most frequent first.
    2. Add the genuinely new classes to the registry (``POST
       /curation/classes``), or add phrasing variants as synonyms in the
       active prompt pack (``OP_PROMPT_PACK_PATH``).
    3. Run this script. Every ``<prefix>_unmatched`` item whose
       ``<prefix>_raw_label`` now resolves to an active class (same resolver
       the live labeler uses: exact, normalised, then synonym) is promoted to
       that class with ``class_source='<prefix>_reclassified'``.

``class_validated`` is NOT set — reclassified items are still machine
suggestions for a human to confirm. Human-owned, validated and
``test_holdout`` items are never touched. Idempotent (converted items no
longer match) and resumable (``search_after`` on ``crop_id``; the final
cursor is printed and can be passed back via ``--start-after``).

``--label-prefix`` selects which labeling path's unmatched items to revisit:
it must be the prefix that path writes (``<prefix>_unmatched`` /
``<prefix>_raw_label``). The default ``vlm`` is what ``POST
/curation/vlm/label_batch`` writes; repeat the flag to cover several paths.

    # Count matches, no writes.
    python3 scripts/curation/reclassify_after_registry_growth.py

    # Convert.
    python3 scripts/curation/reclassify_after_registry_growth.py --apply

    # Several labeling paths, smaller pages, resume after an interruption.
    python3 scripts/curation/reclassify_after_registry_growth.py \\
        --label-prefix vlm --label-prefix other --page-size 500 \\
        --start-after crop_000123 --apply

Index names and the registry path come from ``CurationConfig`` (``OP_*``
env); talks to OpenSearch directly, the API does not need to be running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy import AsyncOpenSearch

from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config
from src.services.curation.registry_reclassify import (
    UnmatchedLabelSource,
    active_name_to_id,
    reclassify_unmatched,
)


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
DEFAULT_PREFIX = 'vlm'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger('curation_reclassify_after_registry_growth')


async def _async_main(args: argparse.Namespace) -> int:
    cfg = get_curation_config()
    registry = ClassRegistry(args.registry or cfg.class_registry_path)
    if not active_name_to_id(registry):
        logger.error('registry %s has no active classes — nothing to match', registry.path)
        return 2

    sources = [UnmatchedLabelSource(p) for p in (args.label_prefix or [DEFAULT_PREFIX])]
    start_after = [args.start_after] if args.start_after else None
    client = AsyncOpenSearch(hosts=[args.opensearch_url], use_ssl=False, timeout=300)
    try:
        for source in sources:
            result = await reclassify_unmatched(
                client,
                source=source,
                registry=registry,
                config=cfg,
                page_size=args.page_size,
                max_pages=args.max_pages,
                search_after=start_after,
                dry_run=args.dry_run,
            )
            summary = result.to_dict()
            print(f'\n[{source.unmatched_source}] ' + json.dumps(summary, indent=2))
        if args.dry_run:
            print('\nDry-run only. Pass --apply to write the reclassifications.')
        return 0
    finally:
        await client.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument(
        '--label-prefix',
        action='append',
        default=None,
        help=f'Labeling-path prefix (repeatable; default {DEFAULT_PREFIX!r}).',
    )
    p.add_argument(
        '--registry',
        type=Path,
        default=None,
        help='Registry JSON (default: CurationConfig.class_registry_path / OP_REGISTRY_PATH).',
    )
    p.add_argument('--page-size', type=int, default=1000, help='Items per search_after page.')
    p.add_argument('--max-pages', type=int, default=0, help='Stop after N pages (0 = no limit).')
    p.add_argument(
        '--start-after',
        default=None,
        help='Resume after this crop_id (the last_cursor printed by a previous run).',
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    args = p.parse_args()
    if args.page_size <= 0:
        p.error('--page-size must be positive')
    if args.start_after and args.label_prefix and len(args.label_prefix) > 1:
        p.error('--start-after resumes a single --label-prefix run')
    return asyncio.run(_async_main(args))


if __name__ == '__main__':
    raise SystemExit(main())

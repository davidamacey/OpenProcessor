#!/usr/bin/env python3
"""Clear the stale class_id/class_name a legacy ``vlm_unmatched`` write left in place.

Finds items with ``class_source='vlm_unmatched'`` and a ``class_id`` still
set -- a write from before the fix in
``src/services/curation/class_sources.py`` (``unmatched_class_clear``) --
and clears the class fields (plus a class-range ``cluster_id``) the same way
the fixed writers now do at write time. A restorable ``class_id_history``
snapshot precedes every write.

Dry run by default (read-only); ``--apply`` writes under OCC.

    python3 scripts/curation/repair_unmatched_class.py
    python3 scripts/curation/repair_unmatched_class.py --verbose
    python3 scripts/curation/repair_unmatched_class.py --apply

The index is ``CurationConfig.items_index`` unless ``--index`` is given.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy import AsyncOpenSearch

from src.config import get_curation_config
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.unmatched_class_repair import RepairPlan, apply_repairs, plan_repairs


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')


def format_plan(plan: RepairPlan) -> str:
    line = (
        f'{plan.crop_id}  class_id={plan.current.get("class_id")!r}  '
        f'cluster_id={plan.current.get("cluster_id")!r}  '
        f'{"clear" if plan.applicable else "(skipped)"}'
    )
    return f'{line}  [{plan.note}]' if plan.note else line


async def run(args: argparse.Namespace, client: object) -> int:
    plans = await plan_repairs(
        client, index=args.index, id_prefix=args.crop_id_prefix, page_size=args.page_size
    )
    if args.verbose:
        for plan in plans:
            print(format_plan(plan))
    applicable = sum(1 for p in plans if p.applicable)
    class_range = sum(
        1
        for p in plans
        if p.applicable
        and isinstance(p.current.get('cluster_id'), int)
        and 0 <= p.current['cluster_id'] < RESIDUAL_CLUSTER_ID_OFFSET
    )
    print(
        f'{len(plans)} candidate(s); {applicable} repairable '
        f'({class_range} in a class-range cluster), {len(plans) - applicable} skipped (locked)'
    )
    for plan in (p for p in plans if not p.applicable):
        print(f'  skipped: {format_plan(plan)}')
    if args.dry_run:
        print('Dry-run only. Pass --apply to write.')
        return 0
    written = await apply_repairs(client, plans, index=args.index)
    print(' '.join(f'{k}={v}' for k, v in written.items()))
    return 1 if written['errors'] else 0


async def _async_main(args: argparse.Namespace) -> int:
    client = AsyncOpenSearch(hosts=[args.opensearch_url], use_ssl=False, timeout=300)
    try:
        return await run(args, client)
    finally:
        await client.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--index', default=get_curation_config().items_index)
    p.add_argument('--crop-id-prefix', default=None, help='Only crop ids starting with this.')
    p.add_argument('--page-size', type=int, default=500)
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument('--verbose', action='store_true', help='Print one line per candidate.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.page_size <= 0:
        build_parser().error('--page-size must be positive')
    return asyncio.run(_async_main(args))


if __name__ == '__main__':
    sys.exit(main())

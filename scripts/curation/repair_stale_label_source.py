#!/usr/bin/env python3
"""Clear a stale ``label_source`` left on a class-less item (K3).

Finds items with ``class_id`` missing but a non-null ``label_source`` --
"who labeled this" recorded with no label to attribute -- and clears
``label_source`` to ``null``. ``class_source`` and every other field are
left untouched: this is narrower than
``scripts/curation/repair_empty_vlm_answers.py`` (which restores full
prior provenance for one specific pattern) and generalizes to any
class-less item carrying a stale ``label_source``, whatever writer left
it there. A human-owned or validated item is never touched.

Dry run by default (read-only); ``--apply`` writes under OCC and records
a restorable ``class_id_history`` snapshot per item.

    python3 scripts/curation/repair_stale_label_source.py
    python3 scripts/curation/repair_stale_label_source.py --verbose
    python3 scripts/curation/repair_stale_label_source.py --apply

The index is ``CurationConfig.items_index`` unless ``--index`` is given.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy import AsyncOpenSearch

from src.config import get_curation_config
from src.services.curation.stale_label_source_repair import RepairPlan, apply_repairs, plan_repairs


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')


def format_plan(plan: RepairPlan) -> str:
    return f'{plan.crop_id}  class_source={plan.prior_class_source!r}  label_source={plan.label_source!r} -> null'


async def run(args: argparse.Namespace, client: object) -> int:
    plans = await plan_repairs(
        client,
        index=args.index,
        id_prefix=args.crop_id_prefix,
        page_size=args.page_size,
    )
    if args.verbose:
        for plan in plans:
            print(format_plan(plan))
    by_class_source: Counter[str] = Counter(str(p.prior_class_source) for p in plans)
    print(f'{len(plans)} candidate(s) with class_id missing + a stale label_source')
    for key, n in by_class_source.most_common():
        print(f'  class_source={key:<30} {n}')
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

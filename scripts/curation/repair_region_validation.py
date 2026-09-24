#!/usr/bin/env python3
"""Move machine-set region validation to ``region_auto_confirmed``.

Older detection workers stamped ``region_validated=true`` when their
auto-confirm policy accepted a box -- no human involved -- which hid those
regions from the human review queue and counted them as human ground
truth. This finds validated rows with no human verdict and rewrites them to
``region_validated=false, region_auto_confirmed=true`` (see
``src/services/curation/region_validation_repair.py``). Human-validated
rows are never touched.

Dry run by default (read-only); ``--apply`` writes under OCC.

    python3 scripts/curation/repair_region_validation.py
    python3 scripts/curation/repair_region_validation.py --apply

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
from src.services.curation.region_validation_repair import (
    apply_region_validation_repair,
    plan_region_validation_repair,
)


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')


async def run(args: argparse.Namespace, client: object) -> int:
    plan = await plan_region_validation_repair(client, index=args.index, page_size=args.page_size)
    total = len(plan.machine_ids) + plan.human_kept
    print(
        f'{total} validated region(s): {len(plan.machine_ids)} machine-set, {plan.human_kept} human'
    )
    for name, counts in (('by_status', plan.by_status), ('by_detector', plan.by_detector)):
        print(f'  machine-set {name}:')
        for key, n in counts.most_common():
            print(f'    {key:<40} {n}')
    if args.dry_run:
        print('Dry-run only. Pass --apply to write.')
        return 0
    result = await apply_region_validation_repair(client, plan, index=args.index)
    errors = result.get('errors') or []
    print(
        f'updated={result.get("updated", 0)} '
        f'skipped_changed={result.get("skipped_due_to_conflict", 0)} errors={len(errors)}'
    )
    return 1 if errors else 0


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
    p.add_argument('--page-size', type=int, default=500)
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
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

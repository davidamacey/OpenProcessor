#!/usr/bin/env python3
"""Restore region detector provenance overwritten by a same-box human confirm.

A human "confirm" (``PUT /crops/{id}/region`` with the box unchanged) used
to re-stamp ``region_detector`` as the human and ``region_score`` as 1.0.
This finds items whose region detector is the human but whose detector
chain records a model hit, and restores the detector, version, score and
detection time from the best source on record — the item's region edit
history, then ``--backup-index`` (the same doc in an index snapshot taken
before the confirm). Items only the detector chain can speak for are
reported but never written: the chain has no score and no pre-confirm box.

Dry run by default (read-only); ``--apply`` writes under OCC and records a
region undo snapshot per item (``POST /crops/{id}/region/undo`` reverts).

    python3 scripts/curation/restore_region_provenance.py
    python3 scripts/curation/restore_region_provenance.py --crop-id-prefix 2b8f1f7f \\
        --backup-index items_backup
    python3 scripts/curation/restore_region_provenance.py --backup-index items_backup --apply

Field names follow ``RegionFields`` (``OP_REGION_FIELD_*``); the index is
``CurationConfig.items_index`` unless ``--index`` is given.
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
from src.services.curation.region_provenance_restore import (
    RestorePlan,
    apply_restores,
    plan_restores,
)


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')


def format_plan(plan: RestorePlan) -> str:
    lines = [
        f'{plan.crop_id}  source={plan.source}  '
        f'{"WILL RESTORE" if plan.applicable else "report only"}'
    ]
    lines.extend(
        f'    {f:<28} {plan.current.get(f)!r:>36} -> {plan.restore.get(f)!r}' for f in plan.fields
    )
    if plan.note:
        lines.append(f'    note: {plan.note}')
    return '\n'.join(lines)


async def run(args: argparse.Namespace, client: object) -> int:
    plans = await plan_restores(
        client,
        index=args.index,
        backup_index=args.backup_index,
        id_prefix=args.crop_id_prefix,
        page_size=args.page_size,
    )
    for plan in plans:
        print(format_plan(plan))
    applicable = sum(1 for p in plans if p.applicable)
    print(
        f'\n{len(plans)} candidate(s); {applicable} restorable, {len(plans) - applicable} report-only'
    )
    if args.dry_run:
        print('Dry-run only. Pass --apply to write.')
        return 0
    counts = await apply_restores(client, plans, index=args.index)
    print(' '.join(f'{k}={v}' for k, v in counts.items()))
    return 1 if counts['errors'] else 0


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
    p.add_argument('--backup-index', default=None, help='Pre-confirm snapshot of the index.')
    p.add_argument('--crop-id-prefix', default=None, help='Only crop ids starting with this.')
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

#!/usr/bin/env python3
"""Restore items stamped ``vlm_unmatched`` for an empty VLM class answer.

Finds items with ``class_source='vlm_unmatched'`` and an empty (or missing)
``vlm_raw_class`` -- the VLM reply carried no class, but the writer recorded
it as an unmatched label -- and restores the class source the item had
before that write (see ``src/services/curation/empty_vlm_answer_repair.py``
for how it is recovered), clears the empty write's ``vlm_confidence`` /
empty raw labels, and records the empty attempt the way the fixed writers
do (``vlm_class_attempted_at`` + ``vlm_class_empty_reason``).

Dry run by default (read-only); ``--apply`` writes under OCC and records a
restorable ``class_id_history`` snapshot per item.

    python3 scripts/curation/repair_empty_vlm_answers.py
    python3 scripts/curation/repair_empty_vlm_answers.py --verbose
    python3 scripts/curation/repair_empty_vlm_answers.py --apply

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
from src.services.curation.empty_vlm_answer_repair import (
    RepairPlan,
    apply_repairs,
    plan_repairs,
    summarize,
)


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')


def format_plan(plan: RepairPlan) -> str:
    target = plan.restore.get('class_source')
    line = (
        f'{plan.crop_id}  source={plan.source}  class_id={plan.current.get("class_id")!r}  '
        f'{plan.current.get("class_source")} -> {target if plan.applicable else "(report only)"}'
    )
    return f'{line}  [{plan.note}]' if plan.note else line


async def run(args: argparse.Namespace, client: object) -> int:
    plans = await plan_repairs(
        client,
        index=args.index,
        id_prefix=args.crop_id_prefix,
        page_size=args.page_size,
        record_attempt=args.record_attempt,
    )
    if args.verbose:
        for plan in plans:
            print(format_plan(plan))
    summary = summarize(plans)
    applicable = sum(1 for p in plans if p.applicable)
    print(
        f'{len(plans)} candidate(s); {applicable} repairable, {len(plans) - applicable} report-only'
    )
    for name, counts in summary.items():
        print(f'  {name}:')
        for key, n in counts.most_common():
            print(f'    {key:<40} {n}')
    for plan in (p for p in plans if not p.applicable):
        print(f'  report-only: {format_plan(plan)}')
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
    p.add_argument(
        '--record-attempt',
        action='store_true',
        help='Stamp the empty answer as a VLM attempt (defers the retry); default: retry now.',
    )
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

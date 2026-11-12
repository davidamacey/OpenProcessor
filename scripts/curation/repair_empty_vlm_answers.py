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
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from typing import TYPE_CHECKING

from scripts.curation._repair_cli import build_repair_parser, exit_code, repair_main
from src.services.curation.empty_vlm_answer_repair import (
    RepairPlan,
    apply_repairs,
    plan_repairs,
    summarize,
)


if TYPE_CHECKING:
    import argparse


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
    return exit_code(await apply_repairs(client, plans, index=args.index))


def _add_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--record-attempt',
        action='store_true',
        help='Stamp the empty answer as a VLM attempt (defers the retry); default: retry now.',
    )


def build_parser() -> argparse.ArgumentParser:
    return build_repair_parser(__doc__, _add_options)


def main() -> int:
    return repair_main(build_parser, run, 'op-repair-empty-vlm-answers')


if __name__ == '__main__':
    sys.exit(main())

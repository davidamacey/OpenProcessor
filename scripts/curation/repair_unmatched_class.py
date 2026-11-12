#!/usr/bin/env python3
"""Clear the stale class_id/class_name a legacy ``vlm_unmatched`` write left in place.

Finds items with ``class_source='vlm_unmatched'`` and a ``class_id`` still
set -- a write from before the fix in
``src/services/curation/class_sources.py`` (``unmatched_class_clear``) --
and clears the class fields the same way
the fixed writers now do at write time. A restorable ``class_id_history``
snapshot precedes every write.

Dry run by default (read-only); ``--apply`` writes under OCC.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from typing import TYPE_CHECKING

from scripts.curation._repair_cli import build_repair_parser, exit_code, repair_main
from src.services.curation.unmatched_class_repair import RepairPlan, apply_repairs, plan_repairs


if TYPE_CHECKING:
    import argparse


def format_plan(plan: RepairPlan) -> str:
    line = (
        f'{plan.crop_id}  class_id={plan.current.get("class_id")!r}  '
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
    print(
        f'{len(plans)} candidate(s); {applicable} repairable'
        f', {len(plans) - applicable} skipped (locked)'
    )
    for plan in (p for p in plans if not p.applicable):
        print(f'  skipped: {format_plan(plan)}')
    if args.dry_run:
        print('Dry-run only. Pass --apply to write.')
        return 0
    return exit_code(await apply_repairs(client, plans, index=args.index))


def build_parser() -> argparse.ArgumentParser:
    return build_repair_parser(__doc__)


def main() -> int:
    return repair_main(build_parser, run, 'op-repair-unmatched-class')


if __name__ == '__main__':
    sys.exit(main())

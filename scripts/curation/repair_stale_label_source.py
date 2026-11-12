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
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from typing import TYPE_CHECKING

from scripts.curation._repair_cli import build_repair_parser, exit_code, repair_main
from src.services.curation.stale_label_source_repair import RepairPlan, apply_repairs, plan_repairs


if TYPE_CHECKING:
    import argparse


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
    return exit_code(await apply_repairs(client, plans, index=args.index))


def build_parser() -> argparse.ArgumentParser:
    return build_repair_parser(__doc__)


def main() -> int:
    return repair_main(build_parser, run, 'op-repair-stale-label-source')


if __name__ == '__main__':
    sys.exit(main())

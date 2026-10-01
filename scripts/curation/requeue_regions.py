#!/usr/bin/env python3
"""Re-run the region stage on items parked in a terminal failure status.

A thin client of the unified reprocess
(:func:`~src.services.curation.reprocess.apply_reprocess`, scope ``region``):
the same selection, lock rule and engine ``POST /reprocess`` uses.

Run after something upstream changed that could rescue previously failed
regions: bbox sanity thresholds, a replaced detector engine, a tightened
verify prompt. Re-run items go back into the detection worker's queue;
nothing is re-ingested.

``--missing-status`` (instead of ``--status``) is the one-off backfill for
items that carry no region status at all (ingested before ingest seeded
``pending_detection`` for an active region profile). The worker never
selects such items, so they need this once after enabling a region profile.

The dry run (default) prints the selected cohort broken down by detector
and rejection reason, and how many items the lock rule skips
(``locked_skipped``: human- or import-validated region sets), before
anything moves. ``--apply`` then:

- sets the status to ``pending_detection`` (``--to pending_verification``
  re-verifies the existing rejected boxes instead);
- keeps the prior status in the region ``status_legacy`` field (first
  re-run only) as an audit trail and clears the rejection reason;
- for the default full re-detect, removes every unlocked machine box and
  keeps human-owned and imported ones (``--to pending_verification`` keeps
  every box).

Writes go through the OCC skip-on-conflict writer, so it is safe with the
worker running (a concurrent write wins); re-running is a no-op once the
cohort is drained.

    # What is parked in detection_failed, by detector x reason?
    python3 scripts/curation/requeue_regions.py --status detection_failed

    # Re-run only one detector's sanity-gate rejections.
    python3 scripts/curation/requeue_regions.py --status detection_failed \\
        --detector my_detector --reason aspect_ratio --apply

    # Re-verify boxes the previous verify prompt rejected.
    python3 scripts/curation/requeue_regions.py --status verify_rejected \\
        --to pending_verification --apply

    # Backfill items ingested before ingest seeded a region status (they
    # never reach the worker otherwise). Dry run first, then --apply.
    python3 scripts/curation/requeue_regions.py --missing-status
    python3 scripts/curation/requeue_regions.py --missing-status --apply

``--detector`` / ``--reason`` are repeatable; pass ``'(none)'`` to select
rows with no detector / reason recorded. Field names follow
``RegionFields`` (``OP_REGION_FIELD_*``) and the index ``CurationConfig``
(``OP_*``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402

from src.config import RegionStatus
from src.services.curation.region_requeue import REQUEUE_TARGETS, REQUEUEABLE_STATUSES
from src.services.curation.reprocess import apply_reprocess
from src.services.curation.reprocess_models import (
    ReprocessFilter,
    ReprocessRequest,
    ReprocessScopeResult,
    ReprocessTargets,
)
from src.services.curation.reprocess_targets import ReprocessTargetsError
from src.services.detection.profile_registry import get_active_region_profile
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import add_project_argument, bind_script_project


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger('curation_requeue_regions')


def _print_breakdown(result: ReprocessScopeResult, label: str) -> None:
    print(f'\n{label}: {result.selected:,} items selected\n')
    by_detector: dict[str, list[tuple[str, int]]] = {}
    for row in result.breakdown:
        by_detector.setdefault(row.detector, []).append((row.reason, row.count))
    for detector, reasons in by_detector.items():
        print(f'  detector={detector:<32} {sum(c for _, c in reasons):>10,}')
        for reason, count in reasons:
            print(f'      reason={reason:<40} {count:>10,}')
    # An item with zero `region_boxes` elements is invisible to the nested
    # detector/reason breakdown above: call it out explicitly.
    no_box = result.detail.get('no_box', 0)
    if no_box:
        print(f'  {"(no box at all)":<41} {no_box:>10,}')
    if result.locked_skipped:
        print(f'  {"(locked, skipped)":<41} {result.locked_skipped:>10,}')


async def _async_main(args: argparse.Namespace, request: ReprocessRequest) -> int:
    client = make_script_opensearch([args.opensearch_url], use_ssl=False, timeout=300)
    label = (
        '(no status) -> ' + args.target
        if args.missing_status
        else f'{args.status} -> {args.target}'
    )
    try:
        try:
            plan = await apply_reprocess(client, request.model_copy(update={'dry_run': True}))
        except ReprocessTargetsError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 2
        result = plan.scopes[0]
        _print_breakdown(result, label)
        if get_active_region_profile() is None:
            print(
                '\nNote: no region profile is configured (OP_REGION_PROFILE); the '
                'detection worker idles and will not process re-run items.'
            )
        if args.dry_run:
            print('\nDry-run only. Pass --apply to re-run.')
            return 0
        if result.selected - result.locked_skipped == 0:
            return 0
        applied = await apply_reprocess(client, request)
        print(f'\nqueued={applied.scopes[0].queued:,}')
        return 0
    finally:
        await client.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--status',
        choices=[s.value for s in REQUEUEABLE_STATUSES],
        help='Terminal status to re-run from.',
    )
    source.add_argument(
        '--missing-status',
        action='store_true',
        help='Select items with no region status at all (pre-seeding backfill).',
    )
    p.add_argument(
        '--to',
        dest='target',
        default=RegionStatus.PENDING_DETECTION.value,
        choices=[s.value for s in REQUEUE_TARGETS],
        help='Pending status to re-run to (default: pending_detection).',
    )
    p.add_argument('--detector', action='append', default=[], help='Filter (repeatable).')
    p.add_argument('--reason', action='append', default=[], help='Filter (repeatable).')
    p.add_argument(
        '--missing-provenance',
        action='store_true',
        help='Only regions with no detector chain recorded.',
    )
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    add_project_argument(p)
    args = p.parse_args()
    bind_script_project(args.project, opensearch_url=args.opensearch_url)

    request = ReprocessRequest(
        targets=ReprocessTargets(
            filter=ReprocessFilter(
                region_status=[] if args.missing_status else [args.status],
                missing_status=args.missing_status,
                detector=args.detector,
                reason=args.reason,
                missing_provenance=args.missing_provenance,
            )
        ),
        scopes=['region'],
        region_mode=(
            'reverify' if args.target == RegionStatus.PENDING_VERIFICATION.value else 'redetect'
        ),
        dry_run=args.dry_run,
    )
    return asyncio.run(_async_main(args, request))


if __name__ == '__main__':
    raise SystemExit(main())

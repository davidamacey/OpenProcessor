#!/usr/bin/env python3
"""Requeue regions parked in a terminal failure status back to pending.

Run after something upstream changed that could rescue previously failed
regions — bbox sanity thresholds, a replaced detector engine, a tightened
verify prompt, a removed pre-filter stage. Requeued items go back into the
detection worker's queue; nothing is re-ingested.

``--missing-status`` (instead of ``--status``) is the one-off backfill for
items that carry no region status at all — ingested before ingest seeded
``pending_detection`` for an active region profile. The worker never selects
such items, so they need this once after enabling a region profile.

The dry run (default) prints the selected cohort broken down by detector and
rejection reason, so you can see which model is producing the failures
before requeueing. ``--apply`` then moves them:

- status -> ``--to`` (``pending_detection`` by default);
- the prior status is kept in the region ``status_legacy`` field (first
  requeue only) as an audit trail;
- the rejection reason is cleared;
- with ``--clear-detection`` every box / verify / text / embedding field is
  cleared too, so the cascade starts from scratch (not allowed with
  ``--to pending_verification``, which re-verifies the existing box).

Human-validated regions are never selected. Writes go through the OCC
skip-on-conflict writer, so it is safe with the worker running (a
concurrent write wins); re-running is a no-op once the cohort is drained.

    # What is parked in detection_failed, by detector x reason?
    python3 scripts/curation/requeue_regions.py --status detection_failed

    # Requeue only one detector's sanity-gate rejections.
    python3 scripts/curation/requeue_regions.py --status detection_failed \\
        --detector my_detector --reason aspect_ratio --apply

    # Re-verify boxes the previous verify prompt rejected.
    python3 scripts/curation/requeue_regions.py --status verify_rejected \\
        --to pending_verification --apply

    # Full re-detect of pre-provenance rejects, capped for a partial run.
    python3 scripts/curation/requeue_regions.py --status verify_rejected \\
        --missing-provenance --clear-detection --max-docs 5000 --apply

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
from opensearchpy import AsyncOpenSearch

from src.config import RegionStatus, get_curation_config
from src.services.curation.region_requeue import (
    REQUEUE_TARGETS,
    REQUEUEABLE_STATUSES,
    RequeueSelection,
    apply_requeue,
    requeue_breakdown,
)
from src.services.detection.profile_registry import get_active_region_profile


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger('curation_requeue_regions')


def _print_breakdown(report: dict) -> None:
    print(f'\n{report["status"]} -> {report["target"]}: {report["total"]:,} regions selected\n')
    for det in report['by_detector']:
        print(f'  detector={det["detector"]:<32} {det["count"]:>10,}')
        for r in det['reasons']:
            print(f'      reason={r["reason"]:<40} {r["count"]:>10,}')


async def _async_main(args: argparse.Namespace, sel: RequeueSelection) -> int:
    cfg = get_curation_config()
    client = AsyncOpenSearch(hosts=[args.opensearch_url], use_ssl=False, timeout=300)
    try:
        report = await requeue_breakdown(client, sel, config=cfg)
        _print_breakdown(report)
        if get_active_region_profile() is None:
            print(
                '\nNote: no region profile is configured (OP_REGION_PROFILE); the '
                'detection worker idles and will not process requeued items.'
            )
        if args.dry_run:
            print('\nDry-run only. Pass --apply to requeue.')
            return 0
        if report['total'] == 0:
            return 0
        totals = await apply_requeue(
            client,
            sel,
            clear_detection=args.clear_detection,
            config=cfg,
            page_size=args.page_size,
            max_docs=args.max_docs,
        )
        print(
            f'\nrequeued={totals["updated"]:,} skipped_concurrent_write={totals["skipped"]:,}'
            f' errors={totals["errors"]:,}'
        )
        return 1 if totals['errors'] else 0
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
        help='Terminal status to requeue from.',
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
        help='Pending status to requeue to (default: pending_detection).',
    )
    p.add_argument('--detector', action='append', default=[], help='Filter (repeatable).')
    p.add_argument('--reason', action='append', default=[], help='Filter (repeatable).')
    p.add_argument(
        '--missing-provenance',
        action='store_true',
        help='Only regions with no detector chain recorded.',
    )
    p.add_argument(
        '--clear-detection',
        action='store_true',
        help='Also clear box/verify/text/embedding fields for a from-scratch re-detect.',
    )
    p.add_argument('--max-docs', type=int, default=0, help='Cap requeued regions (0 = all).')
    p.add_argument('--page-size', type=int, default=500)
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    args = p.parse_args()

    if args.clear_detection and args.target == RegionStatus.PENDING_VERIFICATION.value:
        p.error('--clear-detection drops the box that --to pending_verification re-verifies')
    if args.page_size <= 0 or args.max_docs < 0:
        p.error('--page-size must be positive and --max-docs non-negative')
    sel = RequeueSelection(
        status=None if args.missing_status else RegionStatus(args.status),
        target=RegionStatus(args.target),
        detectors=tuple(args.detector),
        reasons=tuple(args.reason),
        missing_provenance=args.missing_provenance,
    )
    return asyncio.run(_async_main(args, sel))


if __name__ == '__main__':
    raise SystemExit(main())

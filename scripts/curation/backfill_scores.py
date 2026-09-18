#!/usr/bin/env python3
"""Backfill curation scores (uniqueness / near-dup / mistakenness) over the
residual item pool (curation-strategy plan §7 Phase 1).

Thin CLI wrapper over ``src.services.curation.item_scores.job`` — same
argparse + ``--dry-run`` + ``--limit`` conventions as other offline backfill
scripts, but delegates the actual fetch/score/write to the item_scores job
runner so the CLI path and the ``POST /curation/scores/compute`` API path
share identical logic (single embedding fetch, per-scorer bulk writes,
``test_holdout`` exclusion).

Resumable: every scorer overwrites its own fields on the items it processes
(version-stamped — see ``ScoreResult.version`` / plan §4 "Versioning
rule"); there is no partial-item skip logic because each scorer's math
depends on the *whole* residual pool's geometry (k-NN neighbours, IVF
buckets), not just the unscored subset. ``--limit`` caps how many residual
items are pulled (smoke runs only — a k-NN/near-dup score computed over a
truncated pool is not representative of the full one).

    python3 scripts/curation/backfill_scores.py --dry-run
    python3 scripts/curation/backfill_scores.py --apply --scorers uniqueness,near_dup

Requires ``KB_SCORES_ENABLED=1`` (or ``--force``) — same feature-flag gate
as the API path, so an operator can't accidentally backfill scores the
frontend has no way to consume yet.
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

from src.services.curation.item_scores import available_scorers


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger('curation_backfill_scores')


def _scores_enabled() -> bool:
    return os.environ.get('KB_SCORES_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}


async def _async_main(args: argparse.Namespace) -> int:
    valid = set(available_scorers())
    requested = [s.strip() for s in args.scorers.split(',')] if args.scorers else sorted(valid)
    unknown = [s for s in requested if s not in valid]
    if unknown:
        logger.error('unknown scorer(s): %s; valid: %s', unknown, sorted(valid))
        return 1

    if not _scores_enabled() and not args.force:
        logger.error('KB_SCORES_ENABLED is not set — pass --force to backfill anyway')
        return 1

    if args.dry_run:
        print(
            f'\nwould run scorers {requested} over the residual item pool (limit={args.limit or "none"})'
        )
        print('  test_holdout=true items are always excluded (see item_scores.job docstring)')
        return 0

    client = AsyncOpenSearch(hosts=[args.opensearch_url], use_ssl=False, timeout=600)
    try:
        from src.config.curation import IndexRole, get_curation_config, index_name
        from src.services.curation.clustering import embedding_reduce
        from src.services.curation.item_scores import get_scorer
        from src.services.curation.item_scores.job import bulk_write_result

        extra_must = [{'bool': {'must_not': [{'term': {'test_holdout': True}}]}}]
        ids, embeddings = await embedding_reduce.fetch_residual_v6_embeddings_parallel(
            client, extra_must=extra_must
        )
        if args.limit is not None:
            ids, embeddings = ids[: args.limit], embeddings[: args.limit]
        logger.info('fetched %d residual item embeddings', len(ids))

        for name in requested:
            scorer = get_scorer(name)
            result = await scorer.score(ids, embeddings, opensearch=client)
            await bulk_write_result(client, result)
            logger.info('scorer=%s n_scored=%d extra=%s', name, result.n_scored, result.extra)

        index = index_name(get_curation_config(), IndexRole.ITEMS)
        await client.indices.refresh(index=index)
        print(f'\nbackfilled scores for {len(ids):,} residual items: {requested}')
        return 0
    finally:
        await client.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument(
        '--scorers',
        default=None,
        help='Comma-separated scorer ids (default: all — see GET /curation/methods).',
    )
    p.add_argument(
        '--limit', type=int, default=None, help='Cap the residual pool size (smoke runs).'
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='Backfill even if KB_SCORES_ENABLED is unset.',
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--apply', dest='dry_run', action='store_false')
    args = p.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == '__main__':
    raise SystemExit(main())

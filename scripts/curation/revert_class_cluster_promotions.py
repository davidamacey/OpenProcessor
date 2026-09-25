#!/usr/bin/env python3
"""Data repair: revert class-cluster auto-promotions.

Before the aggregation-query fix
(``src/services/curation/clustering/auto_promote.py``), ``auto_promote_clusters``
scored purity over EVERY ``cluster_id`` bucket, including class clusters
(``0 .. RESIDUAL_CLUSTER_ID_OFFSET-1``, where ``cluster_id == class_id`` by
construction). A class cluster's purity is 1.0 by definition -- every member
trivially "agrees" with the cluster because the cluster IS the class -- so
any classifier label with at least ``min_members`` siblings got stamped
``class_validated=true`` from nothing but the classifier's own earlier
output: a circular self-validation, not an independent signal.

This script finds items with that exact signature -- ``class_source ==
'cluster_majority_agreement'`` and the last ``class_id_history`` entry
written by ``auto_promote`` for a ``cluster_id < RESIDUAL_CLUSTER_ID_OFFSET``
-- and restores the prior class assignment from that history entry:
``class_id``, ``class_name``, ``class_source``, ``label_source``,
``confidence`` roll back to what they were immediately before the bad
promotion, and ``class_validated`` is set back to ``false``.

Human-owned items are never touched (``is_human_owned_class`` guard,
mirroring the painless no-op other writers use) -- if a human
subsequently confirmed a class this script would otherwise revert, the
human write wins and the item is left alone.

Defaults to dry-run: prints the cohort size and a sample of affected class
names. Pass ``--apply`` to write.

    # See what would be reverted.
    python3 scripts/curation/revert_class_cluster_promotions.py

    # Actually revert.
    python3 scripts/curation/revert_class_cluster_promotions.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy import AsyncOpenSearch

from src.clients.occ import is_human_owned_class, occ_skip_on_conflict_bulk
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.orchestrator import ITEMS_INDEX
from src.services.curation.ingest_class_sources import CLUSTER_MAJORITY_CLASS_SOURCE


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
_SCROLL_PAGE = 500

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger('revert_class_cluster_promotions')


def _selection_query() -> dict[str, Any]:
    return {
        'bool': {
            'must': [
                {'term': {'class_source': CLUSTER_MAJORITY_CLASS_SOURCE}},
            ],
        },
    }


def _last_auto_promote_entry(source: dict[str, Any]) -> dict[str, Any] | None:
    """The last ``class_id_history`` entry, if it was written by auto_promote
    for a class-range cluster_id. That entry is a snapshot of the item's
    state *before* the bad promotion -- exactly what we want to restore.
    """
    history = source.get('class_id_history') or []
    if not history:
        return None
    last = history[-1]
    if last.get('writer') != 'auto_promote':
        return None
    cluster_id = source.get('cluster_id')
    if not isinstance(cluster_id, int) or cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET:
        return None
    return last


async def _scroll_candidates(client: AsyncOpenSearch, index: str) -> list[dict[str, Any]]:
    """Doc ids + source for every item matching the selection query.

    Fetched with source (not just ids) because the eligibility check
    needs ``class_id_history`` and ``cluster_id``, which the scroll
    doesn't have to pay for twice if we grab them here.
    """
    body = {
        'size': _SCROLL_PAGE,
        'query': _selection_query(),
        '_source': [
            'class_id_history',
            'cluster_id',
            'class_name',
            'class_source',
        ],
    }
    out: list[dict[str, Any]] = []
    resp = await client.search(index=index, body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        out.extend(hits)
        resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:  # nosec B110 — advisory cleanup only
            logger.info('clear_scroll_failed', extra={'error': str(exc)})
    return out


async def _run(opensearch_url: str, *, apply: bool) -> int:
    client = AsyncOpenSearch(hosts=[opensearch_url], use_ssl=False, timeout=300)
    try:
        hits = await _scroll_candidates(client, ITEMS_INDEX)
        eligible: dict[str, dict[str, Any]] = {}
        for h in hits:
            source = h.get('_source') or {}
            entry = _last_auto_promote_entry(source)
            if entry is None:
                continue
            eligible[h['_id']] = entry

        print(
            f'\n{len(hits):,} items carry class_source='
            f'{CLUSTER_MAJORITY_CLASS_SOURCE!r}; {len(eligible):,} were promoted '
            'from a class-range cluster (the aggregation-query bug) and are eligible to revert.\n'
        )
        if eligible:
            restored_names = Counter(e.get('class_name') for e in eligible.values())
            print('Restoring to (top 10 by count):')
            for name, count in restored_names.most_common(10):
                print(f'  {name!r:<30} {count:>8,}')

        if not eligible:
            return 0
        if not apply:
            print('\nDry-run only. Pass --apply to revert.')
            return 0

        def _merge_revert(doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
            if is_human_owned_class(current):
                return {}
            # Re-check freshest state: only revert if it's still the
            # cluster_majority_agreement write we scrolled for.
            if current.get('class_source') != CLUSTER_MAJORITY_CLASS_SOURCE:
                return {}
            entry = eligible.get(doc_id)
            if entry is None:
                return {}
            return {
                'class_id': entry.get('class_id'),
                'class_name': entry.get('class_name'),
                'class_source': entry.get('class_source'),
                'label_source': entry.get('label_source'),
                'confidence': entry.get('confidence'),
                'class_validated': False,
            }

        result = await occ_skip_on_conflict_bulk(
            client,
            doc_ids=list(eligible.keys()),
            merger=_merge_revert,
            index=ITEMS_INDEX,
            refresh=True,
            writer_id='revert_class_cluster_promotions',
        )
        print(
            f'\nreverted={result.get("updated", 0):,} '
            f'skipped_concurrent_write={result.get("skipped_due_to_conflict", 0):,} '
            f'errors={len(result.get("errors", [])):,}'
        )
        return 1 if result.get('errors') else 0
    finally:
        await client.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument('--apply', action='store_true', help='Write reverts (default: dry-run).')
    args = p.parse_args()
    return asyncio.run(_run(args.opensearch_url, apply=args.apply))


if __name__ == '__main__':
    raise SystemExit(main())

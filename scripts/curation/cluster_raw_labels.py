#!/usr/bin/env python3
"""Cluster raw VLM class labels into ranked candidate sub-classes.

Populates the fields ``GET {api_prefix}/review/raw_label_clusters`` reads
(until this runs, that endpoint returns ``status: "empty"``):

1. A terms aggregation pulls every distinct raw VLM label and its item
   count from the items index (optionally only items whose label did not
   resolve to a registry class — ``--unmatched-only``).
2. The labels are embedded and grouped by average-linkage agglomerative
   clustering on cosine distance (see
   :mod:`src.services.curation.raw_label_clusters`); each group is named
   after its most frequent member and ranked by item volume.
3. Every item carrying a raw label gets its cluster id / name / distance
   written back via partial-document bulk updates, paged with
   ``search_after`` on ``crop_id``.

Re-running is safe: cluster ids are a stable hash of the cluster name,
and each run overwrites the three cluster fields on every item it visits.

Embedding backends (``--embedder``):

* ``pe`` — the stack's own PE-Core text encoder, in-process on CPU (needs
  ``torch`` + ``perception_models``; available inside the API container).
* ``sentence-transformers`` — any sentence-transformers model
  (``--st-model``, default ``all-MiniLM-L6-v2``); optional dependency.
* ``hash`` — dependency-free hashed word + character-trigram vectors.
  Collapses spelling/punctuation variants but is not semantic.
* ``auto`` (default) — first of the above that loads, with a warning when
  it falls back to ``hash``.

Usage::

    # Inside the API container (OpenSearch + index names from OP_* env)
    python3 scripts/curation/cluster_raw_labels.py --dry-run --report /tmp/raw_clusters.json
    python3 scripts/curation/cluster_raw_labels.py --min-count 3 --distance-threshold 0.30

    # Only the labels that did not resolve to a registry class
    python3 scripts/curation/cluster_raw_labels.py --unmatched-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.config.curation import CurationConfig
from src.services.curation.raw_label_clusters import (
    CLUSTER_ID_FIELD,
    RAW_LABEL_FIELD,
    UNMATCHED_CLASS_SOURCE,
    ClusterAssignment,
    cluster_terms,
    cluster_update_doc,
    hash_embed,
    rank_clusters,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')

logger = logging.getLogger('cluster_raw_labels')


# =============================================================================
# Embedders
# =============================================================================


def _pe_embedder() -> Callable[[list[str]], np.ndarray]:
    from src.clients.pe_encoder import PEEncoder

    encoder = PEEncoder()
    encoder.warm_text_encoder()
    return encoder.encode_text


def _st_embedder(model_name: str) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    model = SentenceTransformer(model_name)

    def _embed(terms: list[str]) -> np.ndarray:
        return model.encode(terms, batch_size=128, normalize_embeddings=True)

    return _embed


def resolve_embedder(name: str, st_model: str) -> tuple[Callable[[list[str]], np.ndarray], str]:
    """Return ``(embed_fn, backend_label)`` for ``--embedder``."""
    if name == 'hash':
        return hash_embed, 'hash'
    if name == 'pe':
        return _pe_embedder(), 'pe'
    if name == 'sentence-transformers':
        return _st_embedder(st_model), f'sentence-transformers:{st_model}'
    for label, factory in (
        ('pe', _pe_embedder),
        (f'sentence-transformers:{st_model}', lambda: _st_embedder(st_model)),
    ):
        try:
            return factory(), label
        except Exception as exc:
            # Broad on purpose: a missing optional dependency surfaces as
            # ImportError, but a missing/unfetchable checkpoint can raise
            # anything. Either way the next backend is tried and the choice
            # is logged, never silent.
            logger.info('embedder %s unavailable: %s', label, exc)
    logger.warning(
        'no semantic text encoder available; falling back to hashed token/trigram vectors. '
        'Clusters will merge spelling variants only. Install one and re-run for semantic groups.'
    )
    return hash_embed, 'hash'


# =============================================================================
# OpenSearch I/O
# =============================================================================


def _base_query(unmatched_only: bool) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{'exists': {'field': RAW_LABEL_FIELD}}]
    if unmatched_only:
        filters.append({'term': {'class_source': UNMATCHED_CLASS_SOURCE}})
    return {'bool': {'filter': filters}}


async def fetch_label_counts(
    client: Any, index: str, *, max_terms: int, unmatched_only: bool
) -> list[tuple[str, int]]:
    body = {
        'size': 0,
        'query': _base_query(unmatched_only),
        'aggs': {'terms': {'terms': {'field': RAW_LABEL_FIELD, 'size': max_terms}}},
    }
    resp = await client.search(index=index, body=body)
    buckets = (((resp.get('aggregations') or {}).get('terms') or {}).get('buckets')) or []
    return [(str(b['key']), int(b['doc_count'])) for b in buckets if str(b['key']).strip()]


async def write_back(
    client: Any,
    index: str,
    assignments: dict[str, ClusterAssignment],
    *,
    unmatched_only: bool,
    page_size: int,
    dry_run: bool,
) -> dict[str, int]:
    """Walk every item with a raw label and write its cluster fields."""
    stats = {'scanned': 0, 'updated': 0, 'unassigned': 0, 'unchanged': 0, 'errors': 0, 'pages': 0}
    now = datetime.now(UTC).isoformat()
    search_after: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            'size': page_size,
            '_source': ['crop_id', RAW_LABEL_FIELD, CLUSTER_ID_FIELD],
            'query': _base_query(unmatched_only),
            'sort': [{'crop_id': 'asc'}],
        }
        if search_after is not None:
            body['search_after'] = search_after
        resp = await client.search(index=index, body=body)
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            break
        stats['pages'] += 1
        ops: list[dict[str, Any]] = []
        for hit in hits:
            stats['scanned'] += 1
            src = hit.get('_source') or {}
            doc_id = str(src.get('crop_id') or hit.get('_id') or '')
            assignment = assignments.get(str(src.get(RAW_LABEL_FIELD) or '').strip())
            if not doc_id or assignment is None:
                # Labels past --max-terms (or blank) keep whatever cluster
                # fields a previous run wrote.
                stats['unassigned'] += 1
                continue
            # Skip the write if this doc's label-cluster id already
            # matches the fresh assignment — a stable-hash re-cluster of an
            # unchanged label corpus would otherwise rewrite every row.
            if src.get(CLUSTER_ID_FIELD) == int(assignment.cluster_id):
                stats['unchanged'] += 1
                continue
            ops.append({'update': {'_index': index, '_id': doc_id}})
            ops.append({'doc': cluster_update_doc(assignment, now)})
        search_after = hits[-1].get('sort')
        if ops and not dry_run:
            result = await client.bulk(body=ops, refresh=False)
            for item in result.get('items') or []:
                outcome = item.get('update') or {}
                if outcome.get('error') or outcome.get('status') not in (200, 201):
                    stats['errors'] += 1
                else:
                    stats['updated'] += 1
        if search_after is None:
            break
    return stats


# =============================================================================
# Driver
# =============================================================================


async def run(
    client: Any,
    *,
    index: str,
    embed: Callable[[list[str]], np.ndarray],
    max_terms: int = 5000,
    min_count: int = 3,
    distance_threshold: float = 0.30,
    page_size: int = 1000,
    unmatched_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fetch, cluster, and (unless ``dry_run``) write back. Returns a summary."""
    term_counts = await fetch_label_counts(
        client, index, max_terms=max_terms, unmatched_only=unmatched_only
    )
    if not term_counts:
        return {'n_terms': 0, 'n_clusters': 0, 'clusters': [], 'write': {}}
    terms = [t for t, _ in term_counts]
    counts = [c for _, c in term_counts]
    assignments = cluster_terms(
        terms, counts, embed, distance_threshold=distance_threshold, min_count=min_count
    )
    ranked = rank_clusters(assignments, dict(term_counts))

    if not dry_run:
        from src.clients.curation_opensearch import ensure_items_label_cluster_fields

        await ensure_items_label_cluster_fields(client)
    write = await write_back(
        client,
        index,
        assignments,
        unmatched_only=unmatched_only,
        page_size=page_size,
        dry_run=dry_run,
    )
    return {'n_terms': len(terms), 'n_clusters': len(ranked), 'clusters': ranked, 'write': write}


def _print_ranked(ranked: list[dict[str, Any]], top: int) -> None:
    print(f'\n{"cluster_id":>11}  {"n_items":>8}  {"n_terms":>7}  name  [samples]')
    for c in ranked[:top]:
        samples = ', '.join(f'{s["label"]}({s["count"]})' for s in c['sample_terms'])
        print(
            f'{c["cluster_id"]:>11}  {c["n_items"]:>8}  {c["n_terms"]:>7}  '
            f'{c["cluster_name"]}  [{samples}]'
        )


async def _async_main(args: argparse.Namespace) -> int:
    from opensearchpy import AsyncOpenSearch

    index = args.index or CurationConfig.from_env().items_index
    embed, backend = resolve_embedder(args.embedder, args.st_model)
    logger.info('index=%s embedder=%s dry_run=%s', index, backend, args.dry_run)

    client = AsyncOpenSearch(hosts=[args.opensearch_url], use_ssl=False, timeout=600)
    started = time.monotonic()
    try:
        summary = await run(
            client,
            index=index,
            embed=embed,
            max_terms=args.max_terms,
            min_count=args.min_count,
            distance_threshold=args.distance_threshold,
            page_size=args.page_size,
            unmatched_only=args.unmatched_only,
            dry_run=args.dry_run,
        )
    finally:
        await client.close()

    summary['embedder'] = backend
    summary['elapsed_s'] = round(time.monotonic() - started, 1)
    if summary['n_terms'] == 0:
        logger.warning('no %s values in %s; nothing to cluster', RAW_LABEL_FIELD, index)
        return 0
    _print_ranked(summary['clusters'], args.top)
    logger.info(
        '%d terms -> %d clusters; write-back: %s (%.1fs)',
        summary['n_terms'],
        summary['n_clusters'],
        summary['write'],
        summary['elapsed_s'],
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        logger.info('wrote report %s', args.report)
    return 1 if summary['write'].get('errors') else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument('--index', default=None, help='Items index (default: OP_ITEMS_INDEX / config)')
    p.add_argument(
        '--embedder',
        choices=['auto', 'pe', 'sentence-transformers', 'hash'],
        default='auto',
    )
    p.add_argument('--st-model', default='all-MiniLM-L6-v2')
    p.add_argument('--max-terms', type=int, default=5000, help='Distinct labels to cluster')
    p.add_argument(
        '--min-count', type=int, default=3, help='Labels on fewer items stay singleton clusters'
    )
    p.add_argument(
        '--distance-threshold',
        type=float,
        default=0.30,
        help='Cosine-distance cut for average-linkage clustering',
    )
    p.add_argument('--page-size', type=int, default=1000)
    p.add_argument(
        '--unmatched-only',
        action='store_true',
        help='Only cluster labels on items the registry could not resolve',
    )
    p.add_argument('--top', type=int, default=50, help='Clusters to print')
    p.add_argument('--report', type=Path, default=None, help='Write the ranked clusters as JSON')
    p.add_argument('--dry-run', action='store_true', help='Cluster and report; write nothing')
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return asyncio.run(_async_main(build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())

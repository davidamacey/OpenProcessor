#!/usr/bin/env python3
"""LG-1: backfill ``region_embedding`` for existing items.

Nothing in ``main`` writes the per-item PE-Core region embedding yet
(the legacy stack's plate-embedder loop has no equivalent here) --
region-FP clustering and ``/regions/suspected_false_positives`` both
key off it, so with 0 items carrying the field, those paths are
completely inert. This script crops each item's already-accepted
region box out of its source image, encodes it through the same
``pe_image_encoder`` Triton model the rest of the curation stack uses
(:class:`~src.clients.pe_encoder.PEEncoder`), L2-normalizes (the
encoder already does this), and writes the vector back.

Resumable by construction: the selection query excludes items that
already carry the field, so re-running only picks up items ingested
or accepted since the last pass. Land CM-3 before running this for
real -- the region-FP centroid store's distance units were wrong
until that fix, so any FP-clustering pass over freshly-backfilled
vectors would use the wrong thresholds.

Defaults to dry-run: prints the eligible count and a source-image
read failure sample. ``--apply`` writes.

    # See what would be backfilled.
    python3 scripts/curation/backfill_region_embeddings.py

    # Actually backfill.
    python3 scripts/curation/backfill_region_embeddings.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from opensearchpy import AsyncOpenSearch

from scripts.curation.worker.state import _crop_jpeg_from_disk
from src.clients.pe_encoder import PEEncoder
from src.clients.triton_pool import AsyncTritonPool
from src.config import get_curation_config, get_region_fields
from src.services.detection.region_embed import embed_region_crops


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
DEFAULT_TRITON = os.environ.get('TRITON_URL', 'triton-server:8001')
_SCROLL_PAGE = 200
_ENCODE_BATCH = 32

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger('backfill_region_embeddings')


def _selection_query() -> dict[str, Any]:
    F = get_region_fields()
    return {
        'bool': {
            'must': [{'exists': {'field': F.bbox_norm}}],
            'must_not': [{'exists': {'field': F.embedding}}],
        },
    }


async def _scroll_candidates(
    client: AsyncOpenSearch, index: str, *, max_docs: int | None
) -> list[dict[str, Any]]:
    F = get_region_fields()
    body = {
        'size': _SCROLL_PAGE,
        'query': _selection_query(),
        '_source': ['image_path', F.bbox_norm],
    }
    out: list[dict[str, Any]] = []
    resp = await client.search(index=index, body=body, scroll='5m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        out.extend(hits)
        if max_docs is not None and len(out) >= max_docs:
            out = out[:max_docs]
            break
        resp = await client.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:  # nosec B110 — advisory cleanup only
            logger.info('clear_scroll_failed', extra={'error': str(exc)})
    return out


async def _run(
    opensearch_url: str,
    triton_url: str,
    *,
    apply: bool,
    max_docs: int | None,
) -> int:
    F = get_region_fields()
    cfg = get_curation_config()
    client = AsyncOpenSearch(hosts=[opensearch_url], use_ssl=False, timeout=300)
    try:
        hits = await _scroll_candidates(client, cfg.items_index, max_docs=max_docs)
        print(f'\n{len(hits):,} items have a region box but no {F.embedding!r}.\n')
        if not hits:
            return 0
        if not apply:
            print('Dry-run only. Pass --apply to backfill.')
            return 0

        pool = AsyncTritonPool(url=triton_url, pool_size=1, max_concurrent=8)
        await pool.initialize()
        pe = PEEncoder(triton_pool=pool)

        n_written = 0
        n_missing_image = 0
        n_decode_failed = 0
        for start in range(0, len(hits), _ENCODE_BATCH):
            batch = hits[start : start + _ENCODE_BATCH]
            crop_jpegs: list[bytes] = []
            doc_ids: list[str] = []
            for h in batch:
                source = h.get('_source') or {}
                image_path = source.get('image_path')
                bbox = source.get(F.bbox_norm)
                if not image_path or not bbox or len(bbox) != 4:
                    n_missing_image += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in bbox)
                jpeg = await asyncio.to_thread(_crop_jpeg_from_disk, image_path, (x1, y1, x2, y2))
                if jpeg is None:
                    n_missing_image += 1
                    continue
                crop_jpegs.append(jpeg)
                doc_ids.append(h['_id'])

            if not crop_jpegs:
                continue

            embeddings = await embed_region_crops(pe, crop_jpegs)
            bulk: list[dict[str, Any]] = []
            for doc_id, emb in zip(doc_ids, embeddings, strict=True):
                if emb is None:
                    n_decode_failed += 1
                    continue
                bulk.append({'update': {'_index': cfg.items_index, '_id': doc_id}})
                bulk.append({'doc': {F.embedding: emb}})
                n_written += 1
            if bulk:
                await client.bulk(body=bulk, refresh=False)

        try:
            await client.indices.refresh(index=cfg.items_index)
        except Exception as exc:
            logger.debug('refresh_failed: %s', exc)

        print(
            f'\nwritten={n_written:,} missing_source_image={n_missing_image:,} '
            f'decode_failed={n_decode_failed:,}'
        )
        return 0
    finally:
        await client.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--opensearch-url', default=DEFAULT_OPENSEARCH)
    p.add_argument('--triton-url', default=DEFAULT_TRITON)
    p.add_argument('--apply', action='store_true', help='Write embeddings (default: dry-run).')
    p.add_argument('--max-docs', type=int, default=None, help='Cap the cohort (testing).')
    args = p.parse_args()
    return asyncio.run(
        _run(args.opensearch_url, args.triton_url, apply=args.apply, max_docs=args.max_docs)
    )


if __name__ == '__main__':
    raise SystemExit(main())

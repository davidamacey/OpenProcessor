#!/usr/bin/env python3
"""Backfill ``region_embedding`` for existing items.

Nothing in ``main`` writes the per-item PE-Core region embedding yet
(the legacy stack's plate-embedder loop has no equivalent here) --
region-FP clustering and ``/regions/suspected_false_positives`` both
key off it, so with 0 items carrying the field, those paths are
completely inert. This script crops each item's already-accepted
region box out of its source image, encodes it through the same
``pe_image_encoder`` Triton model the rest of the curation stack uses
(:class:`~src.clients.pe_encoder.PEEncoder`), L2-normalizes (the
encoder already does this), and writes the vector back.

W8-cleanup port: selection and cropping now read the W8 ``region_boxes``
list (:mod:`src.services.curation.region_boxes`) instead of the retired
item-level ``region_bbox_norm`` scalar, which the current worker no
longer writes. An item can carry more than one ``accepted`` box; this
script still writes ONE item-level ``region_embedding`` (that field
hasn't moved to per-box storage yet -- see Item 5 of the W8-cleanup
plan), so it picks the highest-``score`` accepted box as the item's
representative crop. Interim choice, not a semantic ranking of which
box "matters most" -- a per-box embeddings replacement should embed
every accepted box.

Resumable by construction: the selection query excludes items that
already carry the field, so re-running only picks up items ingested
or accepted since the last pass. Land the region-FP centroid store
distance-units fix before running this for real -- otherwise any
FP-clustering pass over freshly-backfilled vectors would use the
wrong thresholds.

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
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402

from src.clients.pe_encoder import PEEncoder
from src.clients.triton_pool import AsyncTritonPool
from src.config import get_curation_config, get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation.region_boxes import box_query
from src.services.curation.reprocess_embed import EmbedTarget, reembed_items
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import add_project_argument, bind_script_project


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
DEFAULT_TRITON = os.environ.get('TRITON_URL', 'triton-server:8001')
_SCROLL_PAGE = 200

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
)
logger = logging.getLogger('backfill_region_embeddings')


def _selection_query() -> dict[str, Any]:
    """Items still carrying a box worth embedding, that haven't been yet.

    W8-cleanup M5 fix: ``state in ['accepted', 'false_positive']``, not
    ``accepted`` alone. Pre-W8 selection was ``exists
    region_bbox_norm``, and a false-positive item keeps its box (that's
    the whole point of FP status -- it's a hard negative kept for
    training/analysis), so FP items were always selected. The other
    ported W8 readers (``regions.py``, ``regions_fp.py``, ``stats.py``)
    all already treat FP the same way (``state in [accepted,
    false_positive]``); this query silently narrowed to accepted-only
    when it was ported, which starves ``build_region_fp_centroids``
    (status=false_positive AND exists region_embedding) of its inputs --
    the classic hard-negative case (VLM-rejected, human-marked-FP) never
    gets embedded, since the worker's embed stage only runs at
    DETECTED-write time.
    """
    F = get_region_fields()
    return {
        'bool': {
            'must': [
                box_query(
                    {
                        'terms': {
                            f'{F.boxes}.{F.boxes_state}': [
                                'accepted',
                                RegionStatus.FALSE_POSITIVE.value,
                            ],
                        }
                    },
                    F,
                )
            ],
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
        '_source': ['image_path', F.boxes],
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
    client = make_script_opensearch([opensearch_url], use_ssl=False, timeout=300)
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

        # The per-item work is the unified reprocess's `embed` scope
        # (src/services/curation/reprocess_embed.py): same crop, same
        # encoder, same vector, only the region part.
        by_path: dict[str, EmbedTarget] = {}
        for h in hits:
            source = h.get('_source') or {}
            path = source.get('image_path') or ''
            target = by_path.setdefault(
                path, EmbedTarget(image_id=source.get('image_id') or '', image_path=path)
            )
            target.items.append((h['_id'], source))
        counts = await reembed_items(
            client, pe, list(by_path.values()), parts=frozenset({'region'})
        )
        n_written = counts['region_written']
        n_missing_image = counts['missing_image']
        n_decode_failed = counts['decode_failed']

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
    add_project_argument(p)
    args = p.parse_args()
    bind_script_project(args.project, opensearch_url=args.opensearch_url)
    return asyncio.run(
        _run(args.opensearch_url, args.triton_url, apply=args.apply, max_docs=args.max_docs)
    )


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Backfill per-box ``region_box_embeddings`` for existing items.

Region clustering and ``/regions/suspected_false_positives`` key off the
PE-Core vector of each region box. The detection worker writes it for the
boxes it accepts (``scripts/curation/worker/region_embed_stage.py``); this
script catches every embeddable box that has none -- a box a human drew,
an item accepted before the stage existed, a box whose vector went stale
because it was moved (:func:`~src.services.curation.region_box_embeddings.
missing_boxes`). It crops each such box (``accepted`` or ``false_positive``;
a false-positive box is the FP matcher's input) out of its source image,
encodes it through the same ``pe_image_encoder`` Triton model the rest of
the curation stack uses (:class:`~src.clients.pe_encoder.PEEncoder`), and
writes the vectors back with
:func:`~src.services.curation.region_box_embeddings.write_box_embeddings`,
which never touches the box list.

Resumable by construction: the coverage check skips boxes that already
carry a valid vector, so re-running only picks up what is still missing.

Defaults to dry-run: prints the eligible item and box counts. ``--apply``
writes.

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
from src.services.curation.region_box_embeddings import EMBEDDED_BOX_STATES, missing_boxes
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
    """Items holding a box that may need a vector: ``accepted`` or
    ``false_positive``. A false-positive box keeps its geometry (a hard
    negative kept for training and for the FP centroid store), so it is
    embedded like an accepted one; the per-box coverage check (which needs
    the stored vectors' ids) runs in Python on the scrolled hits."""
    F = get_region_fields()
    return {
        'bool': {
            'must': [
                box_query({'terms': {f'{F.boxes}.{F.boxes_state}': list(EMBEDDED_BOX_STATES)}}, F)
            ],
        },
    }


def _source_fields() -> list[str]:
    """``_source`` of a candidate hit: the box list plus only the ids and
    geometry of the stored vectors (never the vectors themselves)."""
    F = get_region_fields()
    return ['image_path', F.boxes, f'{F.box_embeddings}.box_id', f'{F.box_embeddings}.bbox_norm']


async def _scroll_candidates(
    client: AsyncOpenSearch, index: str, *, max_docs: int | None
) -> list[dict[str, Any]]:
    """Items with at least one embeddable box that has no valid vector."""
    body = {'size': _SCROLL_PAGE, 'query': _selection_query(), '_source': _source_fields()}
    out: list[dict[str, Any]] = []
    resp = await client.search(index=index, body=body, scroll='5m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        out.extend(h for h in hits if missing_boxes(h.get('_source') or {}))
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
    cfg = get_curation_config()
    client = make_script_opensearch([opensearch_url], use_ssl=False, timeout=300)
    try:
        hits = await _scroll_candidates(client, cfg.items_index, max_docs=max_docs)
        n_boxes = sum(len(missing_boxes(h.get('_source') or {})) for h in hits)
        print(f'\n{len(hits):,} items have {n_boxes:,} region box(es) with no valid embedding.\n')
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
            client, pe, list(by_path.values()), parts=frozenset({'region'}), only_missing=True
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

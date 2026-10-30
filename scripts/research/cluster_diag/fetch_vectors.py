"""Read-only export of item metadata and stored crop vectors from one project.

    fetch_vectors.py --opensearch-url http://127.0.0.1:9200 --project NAME --out DIR/NAME \\
        [--limit N]

Writes ``NAME.npy`` (float32, one row per item, ordered by ``crop_id``) and
``NAME.meta.jsonl`` (the same rows' metadata; a byte-upload project's rows also carry
``source_identifier``, the client's file path, because its ``image_path`` is a hash name). The project is only searched, through the
guarded script client bound to the project, so the guard refuses any other index. Rows are
sorted by the unique ``crop_id`` so the export is the same for any shard layout or ingest
order.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.config.curation_accessors import images_index, items_index
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import abind_script_project, add_project_argument


VECTOR_FIELD = 'pe_embedding'
META_FIELDS = (
    'crop_id',
    'image_id',
    'image_path',
    'bbox_norm',
    'detector_class_name',
    'detector_confidence',
    'blur_lap_var',
    'crop_area_norm',
    'crop_rank_in_image',
    'cluster_id',
)
PAGE = 1000


async def fetch(
    client: Any, index: str, limit: int | None
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Vectors and metadata of the items that have a vector, ordered by ``crop_id``."""
    body: dict[str, Any] = {
        'size': PAGE,
        '_source': [*META_FIELDS, VECTOR_FIELD],
        'sort': [{'crop_id': 'asc'}],
        'query': {'exists': {'field': VECTOR_FIELD}},
    }
    vectors: list[list[float]] = []
    meta: list[dict[str, Any]] = []
    while limit is None or len(vectors) < limit:
        hits = (await client.search(index=index, body=body))['hits']['hits']
        if not hits:
            break
        for hit in hits:
            source = hit['_source']
            vectors.append(source.pop(VECTOR_FIELD))
            meta.append(source)
        body['search_after'] = hits[-1]['sort']
    if limit is not None:
        vectors, meta = vectors[:limit], meta[:limit]
    return np.asarray(vectors, dtype=np.float32), meta


async def fetch_identifiers(client: Any, index: str) -> dict[str, str]:
    """``image_id -> source_identifier`` of the images a byte upload named (empty otherwise)."""
    body: dict[str, Any] = {
        'size': PAGE,
        '_source': ['image_id', 'source_identifier'],
        'sort': [{'image_id': 'asc'}],
        'query': {'exists': {'field': 'source_identifier'}},
    }
    out: dict[str, str] = {}
    while hits := (await client.search(index=index, body=body))['hits']['hits']:
        out |= {h['_source']['image_id']: h['_source']['source_identifier'] for h in hits}
        body['search_after'] = hits[-1]['sort']
    return out


def write_export(out: Path, vectors: np.ndarray, meta: Sequence[dict[str, Any]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix('.npy'), vectors)
    out.with_suffix('.meta.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in meta))


async def run(args: argparse.Namespace) -> int:
    await abind_script_project(args.project, opensearch_url=args.opensearch_url)
    client = make_script_opensearch([args.opensearch_url], timeout=600)
    try:
        vectors, meta = await fetch(client, items_index(), args.limit)
        identifiers = await fetch_identifiers(client, images_index())
        for row in meta:
            if row['image_id'] in identifiers:
                row['source_identifier'] = identifiers[row['image_id']]
    finally:
        await client.close()
    write_export(args.out, vectors, meta)
    print(f'{len(vectors)} vectors x {vectors.shape[1] if len(vectors) else 0} -> {args.out}')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--opensearch-url', required=True)
    add_project_argument(ap)
    ap.add_argument('--out', type=Path, required=True, help='path stem, without extension')
    ap.add_argument('--limit', type=int, default=None)
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())

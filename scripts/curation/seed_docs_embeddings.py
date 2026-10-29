"""Give a docs-screenshot project's imported items a stand-in embedding vector.

A dataset import stores labels and class clusters but no vectors (the encoder
runs in Triton), and the cluster grid lists only items that have a vector. On
the GPU-less screenshot stack this writes a deterministic unit vector per item,
a class-specific direction plus per-item noise, so the grid, its ordering and the
embedding counts have something to show. **The vectors are synthetic, not
encoder output**; run it only against a throw-away project (it refuses an
index that is not ``verify_``-prefixed), inside the API container::

    docker exec <api> python scripts/curation/seed_docs_embeddings.py --project coco-import
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain.embedding_state import not_embedded_clause  # noqa: E402  # import follows the sys.path setup above
from src.services.projects.guard import make_script_opensearch  # noqa: E402  # import follows the sys.path setup above
from src.services.projects.script_binding import add_project_argument, bind_script_project  # noqa: E402  # import follows the sys.path setup above


NOISE_NORM = 0.35
CHUNK = 200


def _rng(label: str) -> random.Random:
    return random.Random(
        int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'big')
    )  # deterministic fixture data, not security


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def vector_for(class_name: str, crop_id: str, dim: int) -> list[float]:
    """A unit vector near the class's own direction (same class: close, other classes:
    far), different for every item, identical on every run."""
    centroid = _unit([_rng(f'class:{class_name}').gauss(0, 1) for _ in range(dim)])
    noise = _unit([_rng(f'item:{crop_id}').gauss(0, 1) for _ in range(dim)])
    return [
        round(v, 5)
        for v in _unit([c + NOISE_NORM * n for c, n in zip(centroid, noise, strict=True)])
    ]


def pending_query() -> dict[str, Any]:
    return {
        'size': CHUNK,
        'query': not_embedded_clause(),
        '_source': ['class_name'],
    }


async def _seed(args: argparse.Namespace) -> int:
    from src.config import get_curation_config

    cfg = get_curation_config()
    if not cfg.items_index.startswith('verify_'):
        print(f'refusing: {cfg.items_index} is not a verify_-prefixed harness index')
        return 2
    client = make_script_opensearch([args.opensearch_url], timeout=60, max_retries=3)
    done = 0
    try:
        query = pending_query()
        while True:
            hits = (await client.search(index=cfg.items_index, body=query))['hits']['hits']
            if not hits:
                break
            body: list[dict[str, Any]] = []
            for hit in hits:
                name = hit['_source'].get('class_name') or ''
                body.append({'update': {'_index': cfg.items_index, '_id': hit['_id']}})
                body.append(
                    {
                        'doc': {
                            'pe_embedding': vector_for(name, hit['_id'], cfg.encoder_embedding_dim),
                            'embedding_state': 'embedded',
                        }
                    }
                )
            resp = await client.bulk(body=body, refresh=True)
            if resp.get('errors'):
                print('bulk update reported errors')
                return 1
            done += len(hits)
    finally:
        await client.close()
    print(f'embedded {done} items in {cfg.items_index}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--opensearch-url', default=os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
    )
    add_project_argument(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bind_script_project(args.project, opensearch_url=args.opensearch_url)
    return asyncio.run(_seed(args))


if __name__ == '__main__':
    raise SystemExit(main())

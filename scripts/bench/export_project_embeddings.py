"""Read-only export of one project's stored crop embeddings to the ``vecdb`` npz format.

The npz holds ``vectors`` (float32, L2-normalised) and ``class_name`` (unicode), the two
arrays ``cluster_bench.py --source vecdb`` and ``cluster_ladder.py`` read. The rows are
shuffled with a fixed seed so that every nested prefix is a random subsample of the project
rather than its first ingest folders. The project is only searched (point in time plus
``search_after``); nothing is written to it.

    export_project_embeddings.py --opensearch-url http://127.0.0.1:9200 \\
        --index op_prj_<slug>__items --out vectors.npz --seed 20261028
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import orjson


FIELD = 'pe_embedding'
CLASS_FIELD = 'detector_class_name'


def export(url: str, index: str, page: int) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    classes: list[str] = []
    with httpx.Client(
        base_url=url, timeout=600, headers={'Content-Type': 'application/json'}
    ) as http:
        pit = http.post(f'/{index}/_search/point_in_time', params={'keep_alive': '30m'}).json()
        body: dict[str, object] = {
            'size': page,
            'pit': {'id': pit['pit_id'], 'keep_alive': '30m'},
            'sort': [{'_shard_doc': 'asc'}],
            'query': {'exists': {'field': FIELD}},
            '_source': [FIELD, CLASS_FIELD],
            'track_total_hits': False,
        }
        try:
            while True:
                raw = http.post('/_search', content=orjson.dumps(body)).content
                reply = orjson.loads(raw)
                if 'hits' not in reply:
                    raise RuntimeError(f'search failed: {reply}')
                hits = reply['hits']['hits']
                if not hits:
                    break
                vectors.append(np.array([h['_source'][FIELD] for h in hits], dtype=np.float32))
                classes.extend(str(h['_source'].get(CLASS_FIELD) or 'unknown') for h in hits)
                body['search_after'] = hits[-1]['sort']
        finally:
            http.request(
                'DELETE', '/_search/point_in_time', content=orjson.dumps({'pit_id': pit['pit_id']})
            )
    return np.vstack(vectors), np.asarray(classes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--opensearch-url', required=True)
    ap.add_argument('--index', required=True, help='the project items index (read only)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--page', type=int, default=2000)
    args = ap.parse_args()
    t0 = time.perf_counter()
    vectors, classes = export(args.opensearch_url, args.index, args.page)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    order = np.random.default_rng(args.seed).permutation(len(vectors))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, vectors=vectors[order], class_name=classes[order])
    print(
        f'exported {len(vectors)} x {vectors.shape[1]} in {time.perf_counter() - t0:.1f} s '
        f'-> {args.out}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())

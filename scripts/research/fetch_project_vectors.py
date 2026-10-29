"""Read-only export of one project's item ids, PE vectors and current cluster ids to an npz.

    fetch_project_vectors.py --opensearch-url http://127.0.0.1:9200 --index op_prj_<slug>__items \\
        --out vectors.npz

Only searches (point in time plus search_after); nothing is written to the project. The npz holds
``item_id``, ``vectors`` (float32, L2-normalised) and ``cluster_id`` (-1 when the item has none),
in the stable order of the search.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import numpy as np


def export(url: str, index: str, page: int = 500) -> dict[str, np.ndarray]:
    ids: list[str] = []
    vecs: list[list[float]] = []
    clusters: list[int] = []
    with httpx.Client(base_url=url, timeout=300) as http:
        pit = http.post(f'/{index}/_search/point_in_time', params={'keep_alive': '5m'}).json()
        body: dict[str, object] = {
            'size': page,
            'pit': {'id': pit['id'], 'keep_alive': '5m'},
            'sort': [{'_shard_doc': 'asc'}],
            'query': {'exists': {'field': 'pe_embedding'}},
            '_source': ['pe_embedding', 'cluster_id'],
            'track_total_hits': False,
        }
        try:
            while True:
                hits = http.post('/_search', json=body).json()['hits']['hits']
                if not hits:
                    break
                for h in hits:
                    ids.append(h['_id'])
                    vecs.append(h['_source']['pe_embedding'])
                    cid = h['_source'].get('cluster_id')
                    clusters.append(-1 if cid is None else int(cid))
                body['search_after'] = hits[-1]['sort']
        finally:
            http.request('DELETE', '/_search/point_in_time', json={'id': pit['id']})
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return {'item_id': np.array(ids), 'vectors': arr, 'cluster_id': np.array(clusters)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--opensearch-url', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = export(args.opensearch_url, args.index)
    np.savez(
        args.out, item_id=data['item_id'], vectors=data['vectors'], cluster_id=data['cluster_id']
    )
    print(f'{len(data["item_id"])} x {data["vectors"].shape[1]} -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

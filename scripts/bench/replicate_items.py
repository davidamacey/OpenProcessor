"""Grow a project's items index to a target size by replicating its real documents (bench only).

Query latency at 100k and 1M items needs an index of that size, and a real 1M-item ingest takes
days of GPU time. This copies the project's real, ingested items: each copy keeps every field
and gets a new document id plus its PE vector perturbed by Gaussian noise of ``--sigma`` per
dimension and re-normalised (default 0.01 gives a cosine of about 0.95 to the original), so the
HNSW graph is not built from exact duplicates. The class mix, filters, document sizes and index
settings are the real ones; the vector geometry is a jittered copy of 1 real set, NOT 1M
independent photos. Any number measured on the result must be labelled that way.

    replicate_items.py --opensearch-url http://127.0.0.1:4627 \\
        --index op_prj_<slug>__items --target-items 1000000 --seed 20261102

The first ``--source-count`` documents (point-in-time order) are the originals; running again
on an already grown index only reads that many, so the result is idempotent in size.
"""

from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np
import orjson


FIELD = 'pe_embedding'
JSON = {'Content-Type': 'application/json'}


def copy_doc(
    doc_id: str, source: dict[str, Any], rep: int | str, noise: np.ndarray, sigma: float
) -> tuple[str, dict[str, Any]]:
    """One replica of ``source``: new id, jittered and re-normalised vector."""
    vec = np.asarray(source[FIELD], dtype=np.float32) + noise * sigma
    vec /= max(float(np.linalg.norm(vec)), 1e-12)
    return f'{doc_id}~r{rep}', {**source, FIELD: vec.tolist()}


def bulk_body(docs: list[tuple[str, dict[str, Any]]]) -> bytes:
    lines: list[bytes] = []
    for doc_id, src in docs:
        lines.append(orjson.dumps({'index': {'_id': doc_id}}))
        lines.append(orjson.dumps(src))
    return b'\n'.join(lines) + b'\n'


def read_originals(http: httpx.Client, index: str, limit: int) -> list[tuple[str, dict[str, Any]]]:
    pit = http.post(f'/{index}/_search/point_in_time', params={'keep_alive': '30m'}).json()
    body: dict[str, Any] = {
        'size': 2000,
        'pit': {'id': pit['pit_id'], 'keep_alive': '30m'},
        'sort': [{'_shard_doc': 'asc'}],
        'query': {'exists': {'field': FIELD}},
        'track_total_hits': False,
    }
    out: list[tuple[str, dict[str, Any]]] = []
    try:
        while len(out) < limit:
            hits = orjson.loads(http.post('/_search', content=orjson.dumps(body)).content)['hits'][
                'hits'
            ]
            if not hits:
                break
            out.extend((h['_id'], h['_source']) for h in hits if '~r' not in h['_id'])
            body['search_after'] = hits[-1]['sort']
    finally:
        http.request(
            'DELETE', '/_search/point_in_time', content=orjson.dumps({'pit_id': pit['pit_id']})
        )
    return out[:limit]


def send(url: str, index: str, payload: bytes) -> int:
    with httpx.Client(base_url=url, timeout=600, headers=JSON) as http:
        reply = orjson.loads(
            http.post(
                f'/{index}/_bulk',
                content=payload,
                headers={'Content-Type': 'application/x-ndjson'},
            ).content
        )
    if reply.get('errors'):
        first = next(i for i in reply['items'] if 'error' in i['index'])
        raise RuntimeError(f'bulk errors, first: {first}')
    return len(reply['items'])


def replicate(a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    with httpx.Client(base_url=a.opensearch_url, timeout=600, headers=JSON) as http:
        originals = read_originals(http, a.index, a.source_count)
        count = http.get(f'/{a.index}/_count').json()['count']
        http.put(f'/{a.index}/_settings', json={'index': {'refresh_interval': '-1'}})
    if not originals:
        raise SystemExit('no embedded items to replicate')
    missing = max(0, a.target_items - count)
    rng = np.random.default_rng(a.seed)
    batches: list[list[tuple[str, dict[str, Any]]]] = []
    made = 0
    for rep in itertools.count(1):
        for lo in range(0, len(originals), a.batch):
            if made >= missing:
                break
            chunk = originals[lo : lo + a.batch][: missing - made]
            noise = rng.standard_normal((len(chunk), len(originals[0][1][FIELD])), np.float32)
            batches.append(
                [
                    copy_doc(i, s, f'{a.seed}.{rep}', n, a.sigma)
                    for (i, s), n in zip(chunk, noise, strict=True)
                ]
            )
            made += len(chunk)
            if len(batches) >= a.workers * 2:
                with ThreadPoolExecutor(a.workers) as pool:
                    list(pool.map(lambda b: send(a.opensearch_url, a.index, bulk_body(b)), batches))
                batches = []
        if made >= missing:
            break
    if batches:
        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(lambda b: send(a.opensearch_url, a.index, bulk_body(b)), batches))
    with httpx.Client(base_url=a.opensearch_url, timeout=3600, headers=JSON) as http:
        http.put(f'/{a.index}/_settings', json={'index': {'refresh_interval': '1s'}})
        http.post(f'/{a.index}/_refresh')
        final = http.get(f'/{a.index}/_count').json()['count']
    return {
        'originals': len(originals),
        'added': made,
        'final_items': final,
        'sigma': a.sigma,
        'seconds': round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--opensearch-url', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--target-items', type=int, required=True)
    ap.add_argument('--source-count', type=int, default=1_000_000)
    ap.add_argument('--sigma', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=20261102)
    ap.add_argument('--batch', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=6)
    print(replicate(ap.parse_args()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

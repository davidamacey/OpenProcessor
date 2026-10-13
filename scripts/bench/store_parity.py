"""Product-stack gate for a vector-index engine change: ANN overlap against exact search.

Two projects ingested from the same images on the same stack, one per engine (for example
faiss with ``ef_construction`` 512 and Lucene with 128). For each project the script reads the
stored item vectors, then runs the product's own query builder
(:func:`~src.services.curation.semantic_search.build_knn_query` with the review-default filter,
and the class and cluster filters a reviewer adds) against the live index for queries drawn
from the project's items, and compares the engine's top-k with exact cosine top-k computed from
the same stored vectors. It also reports the overlap of the two engines' top-k lists on the
detections both projects hold (matched by content-hashed file name and box, because item ids
carry the project's upload path) and the cosine ``_score`` agreement. Cluster agreement and
purity come from ``parity.py`` captures of the same two projects.

    store_parity.py --opensearch-url http://127.0.0.1:4967 --ref op_prj_a__items \\
        --cand op_prj_b__items --out store_parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.parity_metrics import adjusted_rand_index, normalized_mutual_info, purity
from src.services.curation.semantic_search import _build_filter, build_knn_query


VECTOR = 'pe_embedding'
SOURCE = ['crop_id', 'image_path', 'bbox_norm', 'detector_class_name', 'cluster_id', VECTOR]


def stable_key(item: dict[str, Any]) -> str:
    """The same detection in two projects: the file name (a content hash) and the box."""
    box = ','.join(f'{v:.2f}' for v in item['bbox_norm'])
    return f'{Path(item["image_path"]).name}|{box}'


def scroll(client: httpx.Client, url: str, index: str, query: dict, source: list[str]) -> list:
    resp = client.post(
        f'{url}/{index}/_search?scroll=2m', json={'size': 1000, 'query': query, '_source': source}
    ).raise_for_status()
    out: list[dict[str, Any]] = []
    while hits := resp.json()['hits']['hits']:
        out += [h['_source'] for h in hits]
        resp = client.post(
            f'{url}/_search/scroll',
            json={'scroll': '2m', 'scroll_id': resp.json()['_scroll_id']},
        ).raise_for_status()
    return out


def cluster_agreement(ref: list[dict[str, Any]], cand: list[dict[str, Any]]) -> dict[str, Any]:
    """Cluster ids of the detections both projects hold: agreement and purity against the
    detector class, each project clustered on its own (k-means has run-to-run variation too)."""
    a = {stable_key(i): i for i in ref if i.get('cluster_id') is not None}
    b = {stable_key(i): i for i in cand if i.get('cluster_id') is not None}
    keys = sorted(a.keys() & b.keys())
    if not keys:
        return {'items': 0}
    la = np.array([a[k]['cluster_id'] for k in keys])
    lb = np.array([b[k]['cluster_id'] for k in keys])
    truth = np.array([a[k].get('detector_class_name') or '' for k in keys])
    pa, pb = purity(la, truth), purity(lb, truth)
    return {
        'items': len(keys),
        'ari': adjusted_rand_index(la, lb),
        'nmi': normalized_mutual_info(la, lb),
        'purity_ref': pa,
        'purity_cand': pb,
        'purity_drop_points': 100 * (pa - pb),
    }


def review_default_filter(extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = _build_filter(
        tab=None,
        cluster_id=None,
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=False,
        include_test=False,
    )
    return [*base, *extra]


def ann_topk(
    client: httpx.Client, url: str, index: str, vec: list[float], flt: list[dict[str, Any]], k: int
) -> list[tuple[str, float]]:
    query = build_knn_query(vec, k=k + 1, filter_clause=flt)
    body = {'size': k + 1, 'query': query, '_source': ['crop_id']}
    hits = client.post(f'{url}/{index}/_search', json=body).raise_for_status().json()['hits']
    return [(h['_source']['crop_id'], h['_score']) for h in hits['hits']]


def run_project(
    client: httpx.Client, url: str, index: str, queries: list[str], k: int, filters: dict[str, Any]
) -> dict[str, Any]:
    items = [
        i
        for i in scroll(client, url, index, {'match_all': {}}, SOURCE)
        if i.get(VECTOR) is not None
    ]
    ids = np.array([i['crop_id'] for i in items])
    mat = np.array([i[VECTOR] for i in items], np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    row = {cid: n for n, cid in enumerate(ids)}
    key_of = {i['crop_id']: stable_key(i) for i in items}
    by_key = {v: c for c, v in key_of.items()}
    result: dict[str, Any] = {'items': len(items)}
    for name, flt in filters.items():
        allowed = {
            i['crop_id'] for i in scroll(client, url, index, {'bool': {'filter': flt}}, ['crop_id'])
        }
        mask = np.isin(ids, list(allowed))
        recalls: list[float] = []
        lists: dict[str, list[tuple[str, float]]] = {}
        for qkey in queries:
            cid = by_key.get(qkey)
            if cid is None:
                continue
            q = mat[row[cid]]
            sims = np.where(mask, mat @ q, -np.inf)
            sims[row[cid]] = -np.inf
            exact = {ids[j] for j in np.argsort(-sims)[:k] if np.isfinite(sims[j])}
            ann = [(c, s) for c, s in ann_topk(client, url, index, q.tolist(), flt, k) if c != cid]
            lists[qkey] = [(key_of[c], sc) for c, sc in ann[:k]]
            if exact:
                recalls.append(len({c for c, _ in ann[:k]} & exact) / len(exact))
        result[name] = {
            'selectivity': float(mask.mean()),
            'recall_at_k_vs_exact': float(np.mean(recalls)) if recalls else None,
            'queries': len(recalls),
            'lists': lists,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--opensearch-url', required=True)
    p.add_argument('--ref', required=True, help='items index of the reference engine')
    p.add_argument('--cand', required=True, help='items index of the candidate engine')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--queries', type=int, default=1000)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--class-name', default='person', help='detector class (project + class)')
    p.add_argument('--seed', type=int, default=20261012)
    args = p.parse_args(argv)
    url = args.opensearch_url.rstrip('/')
    with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
        items = scroll(client, url, args.ref, {'match_all': {}}, SOURCE)
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(items), size=min(args.queries, len(items)), replace=False)
        queries = [stable_key(items[int(i)]) for i in pick]
        clusters = [i['cluster_id'] for i in items if i.get('cluster_id') is not None]
        filters: dict[str, Any] = {
            'review_default': review_default_filter([]),
            'project_class': review_default_filter(
                [{'term': {'detector_class_name': args.class_name}}]
            ),
        }
        if clusters:
            top = max(set(clusters), key=clusters.count)
            filters['project_cluster'] = review_default_filter([{'term': {'cluster_id': top}}])
        cluster = cluster_agreement(
            items, scroll(client, url, args.cand, {'match_all': {}}, SOURCE)
        )
        ref = run_project(client, url, args.ref, queries, args.k, filters)
        cand = run_project(client, url, args.cand, queries, args.k, filters)
    table: dict[str, Any] = {}
    for name in filters:
        both = set(ref[name]['lists']) & set(cand[name]['lists'])
        overlap: list[float] = []
        score_delta: list[float] = []
        for key in both:
            a, b = dict(ref[name]['lists'][key]), dict(cand[name]['lists'][key])
            if a:
                overlap.append(len(a.keys() & b.keys()) / len(a))
            score_delta += [abs(a[c] - b[c]) for c in a.keys() & b.keys()]
        table[name] = {
            'selectivity_ref': ref[name]['selectivity'],
            'recall_ref_vs_exact': ref[name]['recall_at_k_vs_exact'],
            'recall_cand_vs_exact': cand[name]['recall_at_k_vs_exact'],
            'overlap_at_k_ref_vs_cand': float(np.mean(overlap)) if overlap else None,
            'score_abs_delta_p99': float(np.percentile(score_delta, 99)) if score_delta else None,
            'queries': len(both),
        }
    doc = {
        'cluster': cluster,
        'k': args.k,
        'items_ref': ref['items'],
        'items_cand': cand['items'],
        'filters': table,
    }
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

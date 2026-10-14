"""OpenSearch write ceiling for ingest-shaped documents (work package WP-S5).

Replays synthetic documents shaped like real ones into fresh indexes with the real mapping and
settings, and reports how fast one node takes them: documents/s, the images/s that equals, bulk
latency, rejected requests, time until the data is searchable, bytes per document and node CPU.

The real mapping, settings and a pool of real documents come from an existing ingest project
(``--items-index`` and ``--images-index``, for example kept with ``baseline_suite.py``). Each
synthetic item document is a real one with a new id and a new 1024-dimension vector: a real
vector plus gaussian noise, renormalized, so the HNSW graph is built over clustered data like a
real project (``--vectors iid`` draws independent unit vectors instead, a harder graph).

An image's worth of work is ``items_per_image`` item documents and one image document, written
to the two indexes in that proportion. Arms (``--arms``) vary the bulk size, the number of
concurrent writers, ``refresh_interval`` and the HNSW ``ef_construction``; the defaults are the
shipped settings. Nothing is changed in OpenSearch except the throw-away ``w0os-*`` indexes,
which are deleted after each run.

    os_write_ceiling.py --opensearch-url http://127.0.0.1:4707 --project w0tp \\
        --items-index <project>__items --images-index <project>__images --out w0_os.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.baseline_report import percentile
from scripts.bench.suite_lib import SCHEMA, summarize_runs
from scripts.bench.suite_probes import cgroup_cpu_s


ARMS: dict[str, dict[str, Any]] = {
    'shipped': {'bulk': 1000, 'writers': 4, 'refresh': 'shipped', 'ef': None, 'vectors': 'real'},
    'bulk500': {'bulk': 500},
    'bulk5000': {'bulk': 5000},
    'writers1': {'writers': 1},
    'writers8': {'writers': 8},
    'writers16': {'writers': 16},
    'refresh_off': {'refresh': '-1'},
    'ef128': {'ef': 128},
    'iid_vectors': {'vectors': 'iid'},
}
INDEX_SETTING_KEYS = (
    'number_of_shards',
    'number_of_replicas',
    'knn',
    'knn.derived_source',
    'max_inner_result_window',
    'merge',
)


def arm_config(name: str) -> dict[str, Any]:
    """An arm's full settings: the shipped arm overridden by the arm's own keys."""
    return {**ARMS['shipped'], **ARMS[name]}


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


def synth_vectors(
    pool: np.ndarray | None, n: int, dim: int, rng: np.random.Generator, noise: float = 0.35
) -> np.ndarray:
    """``n`` unit vectors: noisy copies of ``pool`` rows, or independent draws when no pool.

    ``noise`` 0 returns ``n`` distinct pool rows as they are (every vector real); it needs
    a pool of at least ``n``.
    """
    if pool is None or len(pool) == 0:
        return unit_rows(rng.standard_normal((n, dim), dtype=np.float32))
    if noise == 0:
        if len(pool) < n:
            raise ValueError(f'{n} real vectors asked for, the pool holds {len(pool)}')
        return unit_rows(pool[rng.permutation(len(pool))[:n]])
    base = pool[rng.integers(0, len(pool), size=n)]
    return unit_rows(base + noise * rng.standard_normal(base.shape, dtype=np.float32) / dim**0.5)


def with_ef_construction(mappings: dict[str, Any], ef: int | None) -> dict[str, Any]:
    """A copy of ``mappings`` with every knn_vector's ``ef_construction`` set to ``ef``."""
    out = copy.deepcopy(mappings)
    if ef is None:
        return out

    def walk(props: dict[str, Any]) -> None:
        for spec in props.values():
            if spec.get('type') == 'knn_vector':
                spec['method']['parameters']['ef_construction'] = ef
            walk(spec.get('properties', {}))

    walk(out['properties'])
    return out


def index_body(source: dict[str, Any], refresh: str, ef: int | None) -> dict[str, Any]:
    """Create-index body from a real index's GET response: its settings, mapping and refresh."""
    settings = {k: v for k, v in source['settings']['index'].items() if k in INDEX_SETTING_KEYS}
    if refresh != 'shipped':
        settings['refresh_interval'] = refresh
    return {
        'settings': {'index': settings},
        'mappings': with_ef_construction(source['mappings'], ef),
    }


def make_jobs(n_items: int, n_images: int, bulk: int) -> list[tuple[str, int, int]]:
    """Bulk jobs as (kind, start, stop), item and image jobs interleaved by position in the run."""
    jobs = [('items', a, min(a + bulk, n_items)) for a in range(0, n_items, bulk)]
    jobs += [('images', a, min(a + bulk, n_images)) for a in range(0, n_images, bulk)]
    totals = {'items': max(n_items, 1), 'images': max(n_images, 1)}
    return sorted(jobs, key=lambda j: (j[1] / totals[j[0]], j[0]))


def bulk_body(index: str, kind: str, docs: list[bytes], start: int, stop: int) -> bytes:
    action = b'{"index":{"_index":"' + index.encode() + b'","_id":"%s%d"}}\n'
    return b''.join(action % (kind[:1].encode(), n) + docs[n] + b'\n' for n in range(start, stop))


def build_docs(
    templates: list[dict[str, Any]], vectors: dict[str, np.ndarray], kind: str, n: int
) -> list[bytes]:
    """``n`` serialized documents: template ``i % len`` with fresh ids and the given vectors."""
    out = []
    for i in range(n):
        doc = copy.copy(templates[i % len(templates)])
        doc['crop_id' if kind == 'items' else 'image_id'] = f'{kind[:1]}{i:08d}'
        for field, matrix in vectors.items():
            doc[field] = matrix[i]
        out.append(orjson.dumps(doc, option=orjson.OPT_SERIALIZE_NUMPY))
    return out


def post_bulk(client: httpx.Client, url: str, body: bytes) -> tuple[float, int, int]:
    """(seconds, failed items, 429 retries) for one bulk request, retrying on 429."""
    started = time.monotonic()
    retries = 0
    while True:
        resp = client.post(
            f'{url}/_bulk', content=body, headers={'Content-Type': 'application/x-ndjson'}
        )
        if resp.status_code == 429 and retries < 60:
            retries += 1
            time.sleep(min(0.5 * retries, 5.0))
            continue
        resp.raise_for_status()
        doc = resp.json()
        failed = sum(1 for it in doc['items'] if it['index'].get('status', 500) >= 300)
        return time.monotonic() - started, failed, retries


def run_arm(
    client: httpx.Client,
    args: argparse.Namespace,
    name: str,
    rep: int,
    source: dict[str, dict[str, Any]],
    docs: dict[str, list[bytes]],
) -> dict[str, Any]:
    cfg = arm_config(name)
    url = args.opensearch_url.rstrip('/')
    names = {k: f'w0os-{name}-r{rep}-{k}'.lower() for k in ('items', 'images')}
    for kind, idx in names.items():
        client.put(
            f'{url}/{idx}', json=index_body(source[kind], cfg['refresh'], cfg['ef'])
        ).raise_for_status()
    jobs = make_jobs(len(docs['items']), len(docs['images']), cfg['bulk'])
    bodies = {
        (kind, a): bulk_body(names[kind], kind, docs[kind], a, b) for kind, a, b in jobs
    }  # built before the clock starts
    cpu0 = cgroup_cpu_s(f'{args.project}-opensearch')
    load0 = os.getloadavg()[0]
    started = time.monotonic()
    with ThreadPoolExecutor(cfg['writers']) as pool:
        results = list(pool.map(lambda j: post_bulk(client, url, bodies[(j[0], j[1])]), jobs))
    load_s = time.monotonic() - started
    cpu_load = cgroup_cpu_s(f'{args.project}-opensearch') - cpu0
    if cfg['refresh'] != 'shipped':
        for idx in names.values():
            client.put(
                f'{url}/{idx}/_settings', json={'index': {'refresh_interval': '1s'}}
            ).raise_for_status()
    for idx in names.values():
        client.post(f'{url}/{idx}/_refresh', timeout=600).raise_for_status()
    ready_s = time.monotonic() - started
    for idx in names.values():
        client.post(f'{url}/{idx}/_flush').raise_for_status()
    count, size = settled_totals(client, url, list(names.values()))
    sent = len(docs['items']) + len(docs['images'])
    lat = [t for t, _, _ in results]
    for idx in names.values():
        client.delete(f'{url}/{idx}').raise_for_status()
    return {
        'arm': name,
        'rep': rep,
        'config': cfg,
        'flat': {
            'docs_per_s': sent / load_s,
            'images_per_s_equivalent': len(docs['images']) / load_s,
            'vectors_per_s': (len(docs['items']) + 2 * len(docs['images'])) / load_s,
            'load_s': load_s,
            'searchable_s': ready_s,
            'refresh_tail_s': ready_s - load_s,
            'bulk_p50_s': percentile(lat, 50),
            'bulk_p95_s': percentile(lat, 95),
            'failed_docs': float(sum(f for _, f, _ in results) + (sent - count)),
            'retries_429': float(sum(r for _, _, r in results)),
            'bytes_per_doc': size / max(count, 1),
            'node_cpu_cores_during_load': cpu_load / load_s,
            'load_avg_1m_before': load0,
        },
    }


def settled_totals(client: httpx.Client, url: str, indexes: list[str]) -> tuple[int, int]:
    """(documents, primary store bytes) of the indexes once the size stops changing.

    Index statistics lag the flush for a moment, so a first read can be near zero.
    """
    previous = -1
    for _ in range(30):
        total = {'docs': 0, 'size': 0}
        for idx in indexes:
            prim = client.get(f'{url}/{idx}/_stats/docs,store').json()['_all']['primaries']
            total['docs'] += prim['docs']['count']
            total['size'] += prim['store']['size_in_bytes']
        if total['size'] == previous and total['size'] > 0:
            break
        previous = total['size']
        time.sleep(2)
    return total['docs'], total['size']


def load_templates(
    client: httpx.Client, url: str, index: str, n: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = client.get(f'{url}/{index}').raise_for_status().json()[index]
    resp = client.post(
        f'{url}/{index}/_search?scroll=5m', json={'size': min(n, 1000), 'query': {'match_all': {}}}
    ).raise_for_status()
    docs: list[dict[str, Any]] = []
    while (hits := resp.json()['hits']['hits']) and len(docs) < n:
        docs += [h['_source'] for h in hits]
        resp = client.post(
            f'{url}/_search/scroll',
            json={'scroll': '5m', 'scroll_id': resp.json()['_scroll_id']},
        ).raise_for_status()
    return docs[:n], source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--opensearch-url', required=True)
    p.add_argument('--project', required=True, help='compose project (node CPU is read from it)')
    p.add_argument('--items-index', required=True)
    p.add_argument('--images-index', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--docs', type=int, default=100_000, help='item documents per run')
    p.add_argument('--items-per-image', type=float, default=6.4)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--templates', type=int, default=1000)
    p.add_argument(
        '--noise',
        type=float,
        default=0.35,
        help='gaussian noise on pool vectors; 0 uses distinct real vectors only',
    )
    args = p.parse_args(argv)
    url = args.opensearch_url.rstrip('/')
    rng = np.random.default_rng(args.seed)
    with httpx.Client(timeout=httpx.Timeout(900.0)) as client:
        item_t, item_src = load_templates(client, url, args.items_index, args.templates)
        image_t, image_src = load_templates(client, url, args.images_index, args.templates)
        pool = {
            'pe': np.array([t['pe_embedding'] for t in item_t if 'pe_embedding' in t], np.float32),
            'img_pe': np.array(
                [t['pe_embedding'] for t in image_t if 'pe_embedding' in t], np.float32
            ),
            'img_clip': np.array([t['embedding'] for t in image_t if 'embedding' in t], np.float32),
        }
        n_images = round(args.docs / args.items_per_image)
        docs_by_mode: dict[str, dict[str, list[bytes]]] = {}
        for mode in sorted({arm_config(a)['vectors'] for a in args.arms.split(',')}):

            def _pool(key: str, mode: str = mode) -> np.ndarray | None:
                return None if mode == 'iid' else pool[key]

            docs_by_mode[mode] = {
                'items': build_docs(
                    item_t,
                    {'pe_embedding': synth_vectors(_pool('pe'), args.docs, 1024, rng, args.noise)},
                    'items',
                    args.docs,
                ),
                'images': build_docs(
                    image_t,
                    {
                        'pe_embedding': synth_vectors(
                            _pool('img_pe'), n_images, 1024, rng, args.noise
                        ),
                        'embedding': synth_vectors(
                            _pool('img_clip'), n_images, 512, rng, args.noise
                        ),
                    },
                    'images',
                    n_images,
                ),
            }
        runs: dict[str, list[dict[str, Any]]] = {}
        for rep in range(1, args.reps + 1):
            for name in args.arms.split(','):
                run = run_arm(
                    client,
                    args,
                    name,
                    rep,
                    {'items': item_src, 'images': image_src},
                    docs_by_mode[arm_config(name)['vectors']],
                )
                runs.setdefault(name, []).append(run)
                print(name, rep, {k: round(v, 2) for k, v in run['flat'].items()}, flush=True)
                args.out.write_text(json.dumps({'runs': runs}, default=str))
        doc = {
            'schema': f'{SCHEMA}-os-1',
            'docs_items': args.docs,
            'docs_images': n_images,
            'items_per_image': args.items_per_image,
            'opensearch': client.get(url).json()['version']['number'],
            'arms': {
                n: {
                    'config': arm_config(n),
                    'summary': summarize_runs([r['flat'] for r in rs]),
                    'runs': rs,
                }
                for n, rs in runs.items()
            },
        }
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

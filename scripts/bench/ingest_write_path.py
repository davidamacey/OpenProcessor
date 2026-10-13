"""The ingest write path alone, at offered image rates the GPU cannot offer yet.

Replays real ingest documents (images docs and their items docs, copied from an ingested
project) through the product's :class:`~src.services.curation.ingest_writer.IngestWriter` and
the guarded product client, at open-loop rates of N images/s, and reports the API-side cost of
the write path: process CPU per image, requests per image, per-image latency (p50, p99) and, in
the same process at the same time, the latency of review-default k-NN queries. The ``perimage``
arm is the previous behaviour (one images bulk, one ``mget`` and one items bulk per image, no
limit on writers); the others coalesce.

Vectors are the real ones from the source project. Documents beyond the number of distinct
source images reuse them with a small gaussian perturbation (``--noise``, relative to unit
norm) so the HNSW graph is not built over exact duplicates; the output records how many
documents were exact real, which is the honest label.

    ingest_write_path.py --opensearch-url http://127.0.0.1:4967 --project wpath \\
        --source-items op_prj_a__items --source-images op_prj_a__images --out write_path.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.baseline_report import percentile
from scripts.bench.suite_probes import cgroup_cpu_s
from src.config import get_curation_config
from src.services.curation.ingest_writer import IngestWriter
from src.services.curation.semantic_search import _build_filter, build_knn_query
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import abind_script_project


GUARDS = ('label_source', 'class_source')
ARMS: dict[str, dict[str, Any]] = {
    'perimage': {'max_docs': 1, 'writers': 1_000_000, 'linger_s': 0.0},
    'coalesced': {},
    'coalesced_w8': {'writers': 8},
    'coalesced_bulk500': {'max_docs': 500},
    'coalesced_nolinger': {'linger_s': 0.0},
}
VECTORS = ('pe_embedding', 'embedding')


def load_images(client: httpx.Client, url: str, items_idx: str, images_idx: str) -> list[dict]:
    """Source images, each with its images doc and its items docs (vectors included)."""

    def scroll(index: str) -> list[dict[str, Any]]:
        out, resp = (
            [],
            client.post(
                f'{url}/{index}/_search?scroll=2m', json={'size': 1000, 'query': {'match_all': {}}}
            ).raise_for_status(),
        )
        while hits := resp.json()['hits']['hits']:
            out += [h['_source'] for h in hits]
            resp = client.post(
                f'{url}/_search/scroll',
                json={'scroll': '2m', 'scroll_id': resp.json()['_scroll_id']},
            ).raise_for_status()
        return out

    items_by_image: dict[str, list[dict]] = {}
    for it in scroll(items_idx):
        items_by_image.setdefault(it['image_id'], []).append(it)
    return [
        {'image': img, 'items': items_by_image[img['image_id']]}
        for img in scroll(images_idx)
        if img['image_id'] in items_by_image
    ]


def perturb(doc: dict[str, Any], rng: np.random.Generator, noise: float) -> dict[str, Any]:
    out = dict(doc)
    for field in VECTORS:
        vec = out.get(field)
        if vec is not None and noise > 0:
            v = np.asarray(vec, np.float32)
            v = v + noise * rng.standard_normal(v.shape, dtype=np.float32) / len(v) ** 0.5
            out[field] = (v / np.linalg.norm(v)).tolist()
    return out


class Counting:
    """Counts the requests the client sends, by operation."""

    def __init__(self, client: Any) -> None:
        self.counts: dict[str, int] = {}
        for name in ('bulk', 'mget'):
            setattr(client, name, self._wrap(name, getattr(client, name)))

    def _wrap(self, name: str, fn: Any) -> Any:
        async def inner(*a: Any, **kw: Any) -> Any:
            self.counts[name] = self.counts.get(name, 0) + 1
            return await fn(*a, **kw)

        return inner


async def run_arm(
    client: Any, sources: list[dict], arm: str, rate: float, seconds: float, args: Any
) -> dict[str, Any]:
    cfg = get_curation_config()
    writer = IngestWriter(client, **ARMS[arm])
    counting = Counting(client)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    run_id = uuid.uuid4().hex[:8]
    latencies: list[float] = []
    failures = 0
    exact_real = docs_total = 0
    knn_lat: list[float] = []
    query_vectors = [it['pe_embedding'] for s in sources[:200] for it in s['items'][:1]]
    flt = _build_filter(
        tab=None,
        cluster_id=None,
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=False,
        include_test=False,
    )

    async def one(n: int) -> None:
        nonlocal failures, exact_real, docs_total
        src = sources[n % len(sources)]
        fresh = n >= len(sources)
        image_id = f'{run_id}{n:08d}'
        image_doc = perturb(src['image'], rng, args.noise if fresh else 0.0) | {
            'image_id': image_id
        }
        crops = [
            perturb(it, rng, args.noise if fresh else 0.0)
            | {'crop_id': f'{run_id}{n:08d}{k:03d}', 'image_id': image_id}
            for k, it in enumerate(src['items'])
        ]
        exact_real += 0 if fresh else 1 + len(crops)
        docs_total += 1 + len(crops)
        started = time.perf_counter()
        try:
            res = await writer.write(
                image_doc,
                crops,
                images_index=cfg.images_index,
                items_index=cfg.items_index,
                guards=GUARDS,
            )
            failures += bool(res.refused)
        except Exception:
            failures += 1
        latencies.append(time.perf_counter() - started)

    async def queries(stop: asyncio.Event) -> None:
        while not stop.is_set():
            vec = py_rng.choice(query_vectors)
            started = time.perf_counter()
            await client.search(
                index=cfg.items_index,
                body={'size': 10, 'query': build_knn_query(vec, k=10, filter_clause=flt)},
            )
            knn_lat.append(time.perf_counter() - started)
            await asyncio.sleep(1 / args.qps)

    stop = asyncio.Event()
    qtask = asyncio.create_task(queries(stop))
    node0 = cgroup_cpu_s(f'{args.compose_project}-opensearch')
    cpu0, wall0 = time.process_time(), time.perf_counter()
    tasks, n = [], 0
    while (elapsed := time.perf_counter() - wall0) < seconds:
        due = int(elapsed * rate) + 1
        while n < due:
            tasks.append(asyncio.create_task(one(n)))
            n += 1
        await asyncio.sleep(0.002)
    await asyncio.gather(*tasks)
    cpu = time.process_time() - cpu0
    node_cpu = cgroup_cpu_s(f'{args.compose_project}-opensearch') - node0
    wall = time.perf_counter() - wall0
    stop.set()
    await qtask
    await writer.close()
    return {
        'arm': arm,
        'offered_images_per_s': rate,
        'images': n,
        'achieved_images_per_s': n / wall,
        'docs': docs_total,
        'docs_exact_real': exact_real,
        'failed_images': failures,
        'api_cpu_ms_per_image': 1000 * cpu / n,
        'node_cpu_ms_per_image': 1000 * node_cpu / n,
        'requests_per_image': sum(counting.counts.values()) / n,
        'requests': dict(counting.counts),
        'bulk_docs_mean': float(np.mean(writer.bulk_sizes)) if writer.bulk_sizes else 0.0,
        'write_latency_p50_ms': 1000 * percentile(latencies, 50),
        'write_latency_p99_ms': 1000 * percentile(latencies, 99),
        'knn_p50_ms': 1000 * percentile(knn_lat, 50) if knn_lat else None,
        'knn_p95_ms': 1000 * percentile(knn_lat, 95) if knn_lat else None,
        'knn_queries': len(knn_lat),
    }


async def amain(args: argparse.Namespace) -> int:
    url = args.opensearch_url.rstrip('/')
    with httpx.Client(timeout=httpx.Timeout(600.0)) as http:
        sources = load_images(http, url, args.source_items, args.source_images)
    await abind_script_project(args.project, opensearch_url=url)
    client = make_script_opensearch([url], timeout=60)
    runs: list[dict[str, Any]] = []
    try:
        for rate in [float(r) for r in args.rates.split(',')]:
            for arm in args.arms.split(','):
                run = await run_arm(client, sources, arm, rate, args.seconds, args)
                runs.append(run)
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in run.items()})
                args.out.write_text(json.dumps({'runs': runs}, indent=2))
    finally:
        await client.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--opensearch-url', required=True)
    p.add_argument('--project', required=True, help='scratch project the documents are written to')
    p.add_argument('--source-items', required=True)
    p.add_argument('--source-images', required=True)
    p.add_argument('--compose-project', default='istore', help='node CPU is read from it')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rates', default='25,50,100,200')
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--seconds', type=float, default=30.0)
    p.add_argument('--qps', type=float, default=20.0, help='k-NN queries/s alongside the writes')
    p.add_argument('--noise', type=float, default=0.35)
    p.add_argument('--seed', type=int, default=20261012)
    return asyncio.run(amain(p.parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())

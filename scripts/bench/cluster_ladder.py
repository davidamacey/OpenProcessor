"""One rung of the cluster-pass size ladder: load, probe k-NN, cluster, record, delete.

Run once per size (a fresh process, so the peak resident set is that size's). Every
repetition gets a fresh throw-away project with the first N rows of a ``vecdb`` npz
bulk-loaded into its items index (the dump stays resident; ``rss_base_mb`` is the resident
set just before the pass). It records:

* the on-disk index size (before and after the cluster pass),
* k-NN latency, unfiltered and filtered by one class, p50/p95/p99 over ``--queries``
  queries (client-side wall time, query vectors sampled from the loaded rows),
* ``--reps`` production IVF cluster passes with the stage split of
  ``cluster_plumbing_bench.one_run`` (fetch, fit, write-back, refresh, tail),
* with ``--with-geometry``, the ``cluster_geometry_stage`` the pipeline runs right after the
  pass (``geometry_s``),
* with ``--assign-only``, ``--topup`` more rows loaded after the pass and the refresh
  daemon's ``assign_only_residuals`` over the pool (``assign_only_s``; it reads the whole
  pool, so its cost is the pool's, not the top-up's),
* with ``--reset``, the whole-index cluster-field reset after the pass (``reset_s``),
* ``whole_pass_ms_per_crop``: pass, geometry and assign-only wall over the pool size, and
* the peak resident set of this process during each pass (``VmHWM`` reset just before).

Each throw-away project is deleted through the product lifecycle after its pass.

    cluster_ladder.py --opensearch-url http://127.0.0.1:9200 --data vectors.npz \\
        --size 50000 --reps 3 --out DIR --state-dir DIR/state
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from scripts.bench import cluster_plumbing_bench as plumbing
from scripts.bench.cluster_bench import delete_throwaway, git_commit
from scripts.bench.cluster_plumbing_bench import (
    FIELD,
    ensure_project,
    fingerprint,
    load_vectors,
    one_run,
    spread,
)
from src.config.curation_accessors import items_index
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import abind_script_project


BULK = 1000
# The real whole-index reset; ``run`` swaps ``plumbing.reset_clusters`` for a no-op around the passes.
reset_clusters = plumbing.reset_clusters
LOAD_CONCURRENCY = 6


def percentiles(ms: list[float]) -> dict[str, float]:
    arr = np.asarray(ms)
    p50, p95, p99 = (float(np.percentile(arr, q)) for q in (50, 95, 99))
    return {'p50_ms': round(p50, 2), 'p95_ms': round(p95, 2), 'p99_ms': round(p99, 2)}


def reset_peak_rss() -> None:
    """Reset ``VmHWM`` so the next peak is the cluster passes' alone (Linux, own process)."""
    Path('/proc/self/clear_refs').write_text('5')


def rss_mb() -> dict[str, float]:
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS', 'VmHWM')):
            fields[line.split(':')[0]] = round(int(line.split()[1]) / 1024, 1)
    return fields


async def load_concurrent(
    client: Any, vectors: np.ndarray, classes: np.ndarray, *, first_id: int = 0
) -> float:
    t0 = time.perf_counter()
    index = items_index()
    sem = asyncio.Semaphore(LOAD_CONCURRENCY)

    async def send(start: int) -> None:
        body: list[dict[str, Any]] = []
        for i in range(start, min(start + BULK, len(vectors))):
            body.append({'index': {'_index': index, '_id': f'b{first_id + i:07d}'}})
            body.append({FIELD: vectors[i].tolist(), 'class_name': str(classes[i])})
        async with sem:
            resp = await client.bulk(body=body, refresh=False)
        if resp.get('errors'):
            raise RuntimeError(f'bulk load failed: {resp["items"][0]}')

    await asyncio.gather(*(send(s) for s in range(0, len(vectors), BULK)))
    await client.indices.refresh(index=index, request_timeout=1800)
    return time.perf_counter() - t0


def index_bytes(url: str) -> dict[str, int]:
    # Plain HTTP: the project guard (rightly) allows no stats request shape.
    stats = httpx.get(f'{url}/{items_index()}/_stats/store,docs', timeout=120).json()
    total = stats['_all']['primaries']
    return {'store_bytes': total['store']['size_in_bytes'], 'docs': total['docs']['count']}


async def knn_latency(
    client: Any, vectors: np.ndarray, classes: np.ndarray, n_queries: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    names, counts = np.unique(classes, return_counts=True)
    # The filtered probe uses the class whose share is nearest 5 percent of the rows.
    target = names[int(np.argmin(np.abs(counts / len(classes) - 0.05)))]
    share = float(counts[names == target][0] / len(classes))
    picks = rng.choice(len(vectors), size=n_queries + 20, replace=False)
    out: dict[str, Any] = {'filter_class_share': round(share, 4)}
    for label, flt in (('unfiltered', None), ('filtered', {'term': {'class_name': str(target)}})):
        times: list[float] = []
        for j, idx in enumerate(picks):
            knn: dict[str, Any] = {'vector': vectors[idx].tolist(), 'k': 10}
            if flt:
                knn['filter'] = flt
            body = {'size': 10, '_source': False, 'query': {'knn': {FIELD: knn}}}
            t0 = time.perf_counter()
            await client.search(index=items_index(), body=body)
            if j >= 20:  # the first 20 warm the page cache and the JIT
                times.append((time.perf_counter() - t0) * 1000)
        out[label] = percentiles(times)
    return out


def whole_pass_ms_per_crop(row: dict[str, Any], size: int) -> float:
    """Wall of the pass plus the stages that follow it (geometry, assign-only) per crop, in ms."""
    total = row['wall_s'] + row.get('geometry_s', 0.0) + row.get('assign_only_s', 0.0)
    return float(round(total / size * 1000, 3))


async def no_reset(_client: Any) -> None:
    """A freshly loaded project has no cluster fields: nothing to reset.

    ``update_by_query`` over the whole index rewrites every document with its vector (a
    graph re-insert each); at 250k rows it ran past half an hour and then failed with a
    client error, so each repetition gets a fresh project instead.
    """


async def one_rep(
    args: argparse.Namespace,
    vectors: np.ndarray,
    classes: np.ndarray,
    topup: tuple[np.ndarray, np.ndarray],
    rep: int,
    attempt: int,
) -> dict[str, Any]:
    slug = f'{args.project_prefix}-{args.size}-{uuid.uuid4().hex[:6]}'
    client = make_script_opensearch([args.opensearch_url], timeout=3600)
    bound = False
    out: dict[str, Any] = {'project': slug}
    try:
        await ensure_project(client, slug)
        await abind_script_project(slug, opensearch_url=args.opensearch_url)
        bound = True
        out['load_s'] = round(await load_concurrent(client, vectors, classes), 1)
        out['index_before'] = index_bytes(args.opensearch_url)
        if rep == 0 and attempt == 1:
            out['knn'] = await knn_latency(client, vectors, classes, args.queries, args.seed)
        base_mb = rss_mb()['VmRSS']
        reset_peak_rss()
        row = await one_run(client, 'ivf', args.size, rep)
        row['rss_mb'] = rss_mb()
        row['rss_base_mb'] = base_mb
        if args.with_geometry:
            row['geometry_s'] = (await plumbing.geometry_run(client, 'ivf', args.size))['wall_s']
        if args.assign_only:
            row['topup'] = len(topup[0])
            await load_concurrent(client, *topup, first_id=args.size)
            again = await plumbing.assign_run(client, args.size, rep, reset=False)
            row['assign_only_s'] = again['wall_s']
            row['assign_only_status'] = again['status']
            row['assign_only_n_assigned'] = again['n_assigned']
            row['assign_only_n_unchanged'] = again['n_unchanged']
        row['whole_pass_ms_per_crop'] = whole_pass_ms_per_crop(row, args.size)
        if args.reset:
            t0 = time.perf_counter()
            await reset_clusters(client)
            row['reset_s'] = round(time.perf_counter() - t0, 1)
        out['row'] = row
        out['index_after'] = index_bytes(args.opensearch_url)
    finally:
        await client.close()
        if bound:
            await delete_throwaway(slug)
    return out


async def run(args: argparse.Namespace, vectors: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    wanted = args.size + (args.topup if args.assign_only else 0)
    if len(vectors) < wanted:
        raise SystemExit(f'the dump has only {len(vectors)} rows, {wanted} needed')
    topup = (vectors[args.size : wanted], classes[args.size : wanted])
    vectors, classes = vectors[: args.size], classes[: args.size]
    plumbing.reset_clusters = no_reset  # type: ignore[assignment]  # one_run's reset hook
    result: dict[str, Any] = {'size': args.size, 'data_sha256': fingerprint(vectors)}
    runs: list[dict[str, Any]] = []
    discarded = 0
    for rep in range(args.reps):
        for attempt in range(1, args.attempts + 1):
            out = await one_rep(args, vectors, classes, topup, rep, attempt)
            row = out['row']
            if row['status'] == 'success' and row['n_residuals'] == args.size:
                break
            # A cluster-refresh daemon on the node assigned some of a new project's items
            # before the fetch, so the pass saw a partial pool: discard it.
            print(f'discarded attempt {attempt}: n_residuals={row["n_residuals"]}', flush=True)
            discarded += 1
        else:
            raise RuntimeError(f'pass did not cluster the pool: {row}')
        row.update(
            attempt=attempt,
            project=out['project'],
            load_s=out['load_s'],
            index_before=out['index_before'],
            index_after=out['index_after'],
        )
        if 'knn' in out:
            result['knn'] = out['knn']
            result['load_s'] = out['load_s']
            result['index_before'] = out['index_before']
            result['index_after'] = out['index_after']
        runs.append(row)
        print(json.dumps(row, default=float), flush=True)
    result['runs'] = runs
    result['discarded_partial_passes'] = discarded
    result['summary'] = {
        k: spread([r[k] for r in runs])
        for k in (
            'wall_s',
            'fetch_s',
            'fit_s',
            'distance_s',
            'write_s',
            'refresh_s',
            'tail_s',
            'whole_pass_ms_per_crop',
            *(('geometry_s',) if args.with_geometry else ()),
            *(('assign_only_s',) if args.assign_only else ()),
        )
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--opensearch-url', required=True)
    ap.add_argument('--data', type=Path, required=True, help='vecdb npz (vectors, class_name)')
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--attempts', type=int, default=3, help='tries per rep (partial pools)')
    ap.add_argument('--with-geometry', action='store_true', help='time cluster_geometry_stage')
    ap.add_argument('--assign-only', action='store_true', help='time assign_only_residuals')
    ap.add_argument('--reset', action='store_true', help='time the whole-index reset (#357)')
    ap.add_argument('--topup', type=int, default=5000, help='rows loaded before assign-only')
    ap.add_argument('--queries', type=int, default=200)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--label', default='run')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--project-prefix', default='clusladder')
    args = ap.parse_args()
    os.environ['OPENSEARCH_URL'] = args.opensearch_url
    os.environ['OP_STATE_DIR'] = str(args.state_dir.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(
        run(
            args,
            *load_vectors('vecdb', args.data, args.size + (args.topup if args.assign_only else 0)),
        )
    )
    result.update(tool='scripts/bench/cluster_ladder.py', commit=git_commit())
    path = args.out / f'{args.label}_{args.size}.json'
    path.write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

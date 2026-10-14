"""Time the production cluster pass at a chosen pool size.

Loads the first N rows of a fixed dump of real PE embeddings of public COCO images into
the items index of a throw-away project (real mapping, through the project factory), runs
the production ``cluster_residuals`` with each requested method and records, per run, the
method, the size, the wall time of the pass and the counts it reports (residuals, clusters).
A pass that did not cluster the whole pool is an error, not a row. The stage split (fetch,
fit, write-back) is ``cluster_plumbing_bench.py``; this tool is the size ladder:

    cluster_bench.py --opensearch-url http://127.0.0.1:9200 \\
        --data arrays.npz --source parity --sizes 2000,10000 --methods ivf --reps 3 \\
        --out DIR

``--data`` is the ``arrays.npz`` of a parity capture (``--source parity``, 12.8k crop
vectors of the pinned COCO set) or an ``export_embeddings`` dump converted to
``vectors``/``class_name`` arrays (``--source vecdb``, 100k and more). The result is one
JSON file in ``--out``. One project per size, named
``<--project-prefix>-<size>-<id>`` (its directories go under ``--state-dir``), is created and
deleted through the product's lifecycle at the end unless ``--keep``. Methods are only called.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.cluster_plumbing_bench import (
    ensure_project,
    fingerprint,
    host_load,
    load_items,
    load_vectors,
    reset_clusters,
)
from src.services.projects.guard import make_script_opensearch
from src.services.projects.script_binding import abind_script_project


class ClusterBenchError(RuntimeError):
    """A pass that did not cluster the whole pool; its time would mean nothing."""


def parse_sizes(raw: str) -> list[int]:
    sizes = sorted({int(s) for s in raw.split(',') if s.strip()})
    if not sizes or sizes[0] <= 0:
        raise ValueError(f'sizes must be positive integers, got {raw!r}')
    return sizes


def take_prefix(pool: np.ndarray, size: int) -> np.ndarray:
    """The first ``size`` rows as unit float32; sizes are nested prefixes of one pool."""
    if size > len(pool):
        raise ValueError(f'size {size} requested but the dump has only {len(pool)} rows')
    head = np.asarray(pool[:size], dtype=np.float32)
    return head / np.maximum(np.linalg.norm(head, axis=1, keepdims=True), 1e-12)


async def timed_pass(
    run_pass: Callable[[str], Awaitable[dict[str, Any]]],
    *,
    method: str,
    size: int,
    rep: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    t0 = clock()
    summary = await run_pass(method)
    wall = clock() - t0
    if summary.get('status') != 'success' or summary.get('n_residuals') != size:
        raise ClusterBenchError(
            f'{method} at {size}: status={summary.get("status")!r} '
            f'n_residuals={summary.get("n_residuals")!r}; the pass did not cluster the pool'
        )
    return {
        'method': method,
        'size': size,
        'rep': rep,
        'wall_s': round(wall, 3),
        'status': summary['status'],
        'n_residuals': summary['n_residuals'],
        'n_clusters': summary.get('n_clusters'),
        'load': host_load(),
    }


def git_commit() -> str:
    try:
        return subprocess.run(  # nosec B603 B607  fixed argv list, no shell, git from PATH
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


async def delete_throwaway(slug: str) -> None:
    """Delete the project through the product's own lifecycle: its indexes, directories and
    registry record. Needs the same OP_STATE_DIR the project was created under."""
    from src.services.projects import lifecycle
    from src.services.projects.delete import delete_project_finish
    from src.services.projects.guard import bind_registry_admin, make_curation_opensearch
    from src.services.projects.registry import get_project_registry

    client = await make_curation_opensearch()
    await get_project_registry().refresh_strict()
    with bind_registry_admin():
        await lifecycle.delete_project(client, slug=slug, confirm=slug)
        await delete_project_finish(client, slug=slug)


async def bench_size(args: argparse.Namespace, size: int, pool: np.ndarray) -> list[dict[str, Any]]:
    from src.services.curation.clustering.backend import detect_cluster_backend
    from src.services.curation.clustering.orchestrator import cluster_residuals

    vectors = take_prefix(pool, size)
    slug = f'{args.project_prefix}-{size}-{uuid.uuid4().hex[:6]}'
    client = make_script_opensearch([args.opensearch_url], timeout=600)
    rows: list[dict[str, Any]] = []
    bound = False
    try:
        await ensure_project(client, slug)
        await abind_script_project(slug, opensearch_url=args.opensearch_url)
        bound = True
        load_s = await load_items(client, vectors, args.truth[:size])
        # The backend probe imports umap/numba once per process; timed runs start warm.
        await asyncio.to_thread(detect_cluster_backend)

        async def run_pass(method: str) -> dict[str, Any]:
            return await cluster_residuals(client, clustering_method=method)

        for method in args.methods.split(','):
            for rep in range(args.reps):
                for attempt in range(1, args.attempts + 1):
                    await reset_clusters(client)
                    try:
                        row = await timed_pass(run_pass, method=method, size=size, rep=rep)
                    except ClusterBenchError as exc:
                        # A cluster-refresh daemon on the same node assigns new projects' items
                        # between the reset and the fetch; the pass saw a partial pool. Discard it.
                        print(f'discarded attempt {attempt}: {exc}', flush=True)
                        if attempt == args.attempts:
                            raise
                        continue
                    break
                rows.append(
                    {**row, 'attempt': attempt, 'project': slug, 'load_s': round(load_s, 2)}
                )
                print(json.dumps(rows[-1]), flush=True)
    finally:
        await client.close()
        if bound and not args.keep:
            await delete_throwaway(slug)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--opensearch-url', required=True)
    ap.add_argument('--data', type=Path, required=True, help='npz of real public embeddings')
    ap.add_argument('--source', choices=['vecdb', 'parity'], default='parity')
    ap.add_argument('--out', type=Path, required=True, help='directory for the result JSON')
    ap.add_argument('--sizes', default='2000,10000')
    ap.add_argument('--methods', default='ivf', help='comma list: ivf, ahc, hdbscan')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument(
        '--attempts',
        type=int,
        default=3,
        help='tries per rep when a pass saw a partial pool (a refresh daemon on the node)',
    )
    ap.add_argument('--project-prefix', default='cwlife-clusbench')
    ap.add_argument(
        '--state-dir',
        type=Path,
        help='project directories of the throw-away projects (default: <out>/state)',
    )
    ap.add_argument('--keep', action='store_true', help='keep the throw-away projects')
    args = ap.parse_args(argv)

    # Project creation goes through the process client, which reads these two settings.
    os.environ['OPENSEARCH_URL'] = args.opensearch_url
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ['OP_STATE_DIR'] = str((args.state_dir or args.out / 'state').resolve())
    sizes = parse_sizes(args.sizes)
    pool, args.truth = load_vectors(args.source, args.data, max(sizes))

    async def all_sizes() -> list[dict[str, Any]]:
        # One event loop: the process-level OpenSearch client is bound to the loop it was made in.
        return [r for size in sizes for r in await bench_size(args, size, pool)]

    rows = asyncio.run(all_sizes())
    result = {
        'tool': 'scripts/bench/cluster_bench.py',
        'commit': git_commit(),
        'data_sha256': fingerprint(pool),
        'source': args.source,
        'methods': args.methods.split(','),
        'rows': rows,
    }
    path = args.out / f'cluster_bench_{int(time.time())}.json'
    path.write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

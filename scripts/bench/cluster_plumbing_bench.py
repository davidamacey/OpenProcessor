"""Where the cluster pass spends its time, stage by stage (fetch, fit, write-back).

Loads the first N rows of a fixed real embedding dump into the items index of a
throw-away project (real mapping, through the project factory), then runs the
production ``cluster_residuals`` per method and splits its wall time:

* ``fetch_s``: ``fetch_residual_embeddings_parallel`` (search plus parse plus matrix).
* ``fit_s``: the method's ``fit_predict`` (train and assign; IVF also reports its own).
* ``distance_s``: member-centroid distances for the methods that have none of their own.
* ``write_s``: from the end of the distances to the start of the index refresh (the
  bulk write-back).
* ``refresh_s`` and ``tail_s``: the refresh, and the gate/centroid-store bookkeeping after.

``wall_s`` is "ingest complete" to "clusters available" for the pool. The stage split
is taken by wrapping the production functions from outside, so the same script times
the code before and after a change. The methods are only called, never modified.

``--fetch-profile`` splits the fetch itself, with plain HTTP against the node: server
time per page, response bytes, JSON decode, and building the matrix from the decoded
lists.

    cluster_plumbing_bench.py --opensearch-url http://127.0.0.1:4717 \\
        --data base.npz --out DIR --sizes 2000,10000 --methods ivf,ahc,hdbscan --reps 3

Only real embeddings are used (``--source vecdb``: the 114,987 public-image vectors of
the vector-store evaluation; ``parity``: the crop vectors of a parity capture).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.config.curation import items_index
from src.services.projects.guard import (
    bind_registry_admin,
    make_curation_opensearch,
    make_script_opensearch,
)
from src.services.projects.script_binding import abind_script_project


LOAD_BULK = 1000
FIELD = 'pe_embedding'


def load_vectors(source: str, path: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit float32 rows and their class names; the sizes are nested prefixes."""
    z = np.load(path)
    if source == 'vecdb':
        vectors, truth = z['vectors'], z['class_name'].astype(str)
    elif source == 'parity':
        keep = z['item_has_vec']
        vectors, truth = z['item_vec'][keep], z['item_gt'][keep].astype(str)
    else:
        raise ValueError(f'unknown source {source!r}')
    vectors = np.asarray(vectors[:limit], dtype=np.float32)
    norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return (vectors / norms).astype(np.float32), truth[:limit]


def fingerprint(vectors: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vectors, dtype=np.float32).tobytes()).hexdigest()


def spread(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {'median': float(np.median(arr)), 'min': float(arr.min()), 'max': float(arr.max())}


async def ensure_project(client: Any, slug: str) -> None:
    from src.services.projects import lifecycle
    from src.services.projects.bootstrap import ensure_projects_index

    # Creation goes through the process client (OPENSEARCH_HOSTS), whose registry the
    # lifecycle keeps current; the script client below only reads and writes items.
    client = await make_curation_opensearch()
    with bind_registry_admin():
        await ensure_projects_index(client)
        try:
            await lifecycle.create_project(client, slug=slug, display_name=slug)
        except Exception as exc:
            if 'slug_taken' not in str(exc):
                raise


async def load_items(client: Any, vectors: np.ndarray, truth: np.ndarray) -> float:
    """Bulk-load the rows as item documents; returns wall seconds."""
    t0 = time.perf_counter()
    index = items_index()
    for start in range(0, len(vectors), LOAD_BULK):
        body: list[dict[str, Any]] = []
        for i in range(start, min(start + LOAD_BULK, len(vectors))):
            body.append({'index': {'_index': index, '_id': f'b{i:07d}'}})
            body.append({FIELD: vectors[i].tolist(), 'class_name': str(truth[i])})
        resp = await client.bulk(body=body, refresh=False)
        if resp.get('errors'):
            raise RuntimeError(f'bulk load failed: {resp["items"][0]}')
    await client.indices.refresh(index=index)
    return time.perf_counter() - t0


async def reset_clusters(client: Any) -> None:
    """Return every document to "fresh from ingest" (no cluster fields)."""
    await client.update_by_query(
        index=items_index(),
        body={
            'query': {'match_all': {}},
            'script': {
                'lang': 'painless',
                'source': "ctx._source.remove('cluster_id'); ctx._source.remove('cluster_subid');"
                " ctx._source.remove('cluster_distance');"
                " ctx._source.remove('cluster_distance_cluster_id');",
            },
        },
        refresh=True,
        wait_for_completion=True,
        conflicts='proceed',
        request_timeout=1800,
    )


class StageClock:
    """Wraps the production stage functions from outside and records their times."""

    def __init__(self, client: Any) -> None:
        self.t: dict[str, float] = {}
        self._client = client
        self._orig: list[tuple[Any, str, Any]] = []

    def _wrap(self, owner: Any, name: str, key: str) -> None:
        orig = getattr(owner, name)
        clock = self

        def record(t0: float) -> None:
            clock.t[key] = clock.t.get(key, 0.0) + time.perf_counter() - t0
            clock.t[key + '_end'] = time.perf_counter()

        def timed(*a: Any, **k: Any) -> Any:
            t0 = time.perf_counter()
            result = orig(*a, **k)
            if not inspect.isawaitable(result):
                record(t0)
                return result

            async def waited() -> Any:
                try:
                    return await result
                finally:
                    record(t0)

            return waited()

        self._orig.append((owner, name, orig))
        setattr(owner, name, timed)

    def __enter__(self) -> StageClock:
        from src.services.curation.clustering import centroid_distance, methods, residual_fetch

        self._wrap(residual_fetch, 'fetch_residual_embeddings_parallel', 'fetch_s')
        self._wrap(centroid_distance, 'member_centroid_distances', 'distance_s')
        self._wrap(self._client.indices, 'refresh', 'refresh_s')
        real_get = methods.get_method
        clock = self

        def get_method(*a: Any, **k: Any) -> Any:
            m = real_get(*a, **k)
            orig = m.fit_predict

            async def fit_predict(*fa: Any, **fk: Any) -> Any:
                t0 = time.perf_counter()
                try:
                    return await orig(*fa, **fk)
                finally:
                    clock.t['fit_s'] = time.perf_counter() - t0
                    clock.t['fit_s_end'] = time.perf_counter()

            m.fit_predict = fit_predict  # type: ignore[method-assign]  # instruments the instance to time the fit
            return m

        self._orig.append((methods, 'get_method', real_get))
        methods.get_method = get_method
        return self

    def __exit__(self, *_: object) -> None:
        for owner, name, orig in reversed(self._orig):
            if owner is self._client.indices:
                owner.__dict__.pop(name, None)
            else:
                setattr(owner, name, orig)


def host_load() -> dict[str, Any]:
    return {'loadavg': [round(x, 2) for x in os.getloadavg()], 'cpus': os.cpu_count()}


async def one_run(client: Any, method: str, size: int, rep: int) -> dict[str, Any]:
    from src.services.curation.clustering.backend import detect_cluster_backend
    from src.services.curation.clustering.orchestrator import cluster_residuals

    # The backend probe imports umap/numba on its first call in a process (about 12 s once,
    # cached after); a long-lived worker pays it once, so the timed runs start warm.
    await asyncio.to_thread(detect_cluster_backend)
    await reset_clusters(client)
    load_before = host_load()
    with StageClock(client) as clock:
        t0 = time.perf_counter()
        summary = await cluster_residuals(client, clustering_method=method)
        t_end = time.perf_counter()
    c = clock.t
    after_fit = c.get('distance_s_end', c.get('fit_s_end', t0))
    refresh_start = c['refresh_s_end'] - c['refresh_s'] if 'refresh_s' in c else t_end
    extra = summary.get('cluster_method_extra') or {}
    row: dict[str, Any] = {
        'kind': 'full',
        'method': method,
        'size': size,
        'rep': rep,
        'status': summary.get('status'),
        'n_residuals': summary.get('n_residuals'),
        'n_clusters': summary.get('n_clusters'),
        'wall_s': round(t_end - t0, 3),
        'fetch_s': round(c.get('fetch_s', 0.0), 3),
        'fit_s': round(c.get('fit_s', 0.0), 3),
        'distance_s': round(c.get('distance_s', 0.0), 3),
        'write_s': round(max(refresh_start - after_fit, 0.0), 3),
        'refresh_s': round(c.get('refresh_s', 0.0), 3),
        'tail_s': round(max(t_end - c.get('refresh_s_end', t_end), 0.0), 3),
        'ivf_train_s': extra.get('train_s'),
        'ivf_assign_s': extra.get('assign_s'),
        'load_before': load_before,
        'load_after': host_load(),
    }
    # The stage that contains the fetch before the fit starts: whatever is left of wall.
    stages = ('fetch_s', 'fit_s', 'distance_s', 'write_s', 'refresh_s', 'tail_s')
    row['other_s'] = round(row['wall_s'] - sum(row[k] for k in stages), 3)
    return row


async def geometry_run(client: Any, method: str, size: int) -> dict[str, Any]:
    """The ``cluster_geometry`` stage the pipeline runs right after the cluster pass, inside
    the same progress stage: it reads every clustered item's vector again, cluster by cluster,
    and rewrites the items, so it counts toward "clusters available"."""
    from src.services.curation.clustering.cluster_geometry import cluster_geometry_stage

    t0 = time.perf_counter()
    summary = await cluster_geometry_stage(client)
    return {
        'kind': 'geometry',
        'method': method,
        'size': size,
        'wall_s': round(time.perf_counter() - t0, 3),
        'summary': {k: v for k, v in (summary or {}).items() if isinstance(v, (int, float, str))},
    }


async def assign_run(client: Any, size: int, rep: int, *, reset: bool = True) -> dict[str, Any]:
    from src.services.curation.clustering.orchestrator import assign_only_residuals

    if reset:
        await reset_clusters(client)
    t0 = time.perf_counter()
    summary = await assign_only_residuals(client)
    return {
        'kind': 'assign_only' if reset else 'assign_only_repeat',
        'n_unchanged': summary.get('n_unchanged'),
        'method': 'ivf',
        'size': size,
        'rep': rep,
        'status': summary.get('status'),
        'n_assigned': summary.get('n_assigned'),
        'wall_s': round(time.perf_counter() - t0, 3),
        'load_after': host_load(),
    }


async def label_digest(client: Any) -> dict[str, Any]:
    """Hash of every (id, cluster_id, distance) so two runs can be compared exactly."""
    from opensearchpy.helpers import async_scan

    rows = []
    async for hit in async_scan(
        client,
        index=items_index(),
        query={'query': {'match_all': {}}, '_source': ['cluster_id', 'cluster_distance']},
        preserve_order=False,
    ):
        s = hit['_source']
        rows.append((hit['_id'], s.get('cluster_id'), s.get('cluster_distance')))
    rows.sort()
    return {'n': len(rows), 'sha256': hashlib.sha256(json.dumps(rows).encode()).hexdigest()}


def fetch_profile(url: str, index: str, pages: int = 5, page_size: int = 2000) -> dict[str, Any]:
    """Time the pieces of one page of the fetch with plain HTTP (no client library)."""
    with httpx.Client(
        base_url=url, timeout=300, headers={'content-type': 'application/json'}
    ) as http:
        pit = http.post(f'/{index}/_search/point_in_time', params={'keep_alive': '5m'}).json()
        body = {
            'size': page_size,
            '_source': [FIELD],
            'query': {'bool': {'filter': [{'exists': {'field': FIELD}}]}},
            'pit': {'id': pit['pit_id'], 'keep_alive': '5m'},
            'sort': [{'_shard_doc': 'asc'}],
        }
        rows: list[dict[str, float]] = []
        for _ in range(pages):
            t0 = time.perf_counter()
            resp = http.post('/_search', content=orjson.dumps(body))
            t1 = time.perf_counter()
            raw = resp.content
            t2 = time.perf_counter()
            decoded_std = json.loads(raw)
            t3 = time.perf_counter()
            decoded = orjson.loads(raw)
            t4 = time.perf_counter()
            hits = decoded['hits']['hits']
            if not hits:
                break
            per_doc = [np.asarray(h['_source'][FIELD], dtype=np.float32) for h in hits]
            np.vstack(per_doc)
            t5 = time.perf_counter()
            np.array([h['_source'][FIELD] for h in hits], dtype=np.float32)
            t6 = time.perf_counter()
            rows.append(
                {
                    'docs': len(hits),
                    'bytes': len(raw),
                    'server_and_transfer_s': t1 - t0,
                    'json_stdlib_s': t3 - t2,
                    'json_orjson_s': t4 - t3,
                    'asarray_per_doc_vstack_s': t5 - t4,
                    'array_whole_page_s': t6 - t5,
                }
            )
            if hits:
                body['search_after'] = hits[-1]['sort']
            del decoded_std
        http.request(
            'DELETE', '/_search/point_in_time', content=orjson.dumps({'pit_id': [pit['pit_id']]})
        )
    keys = [k for k in rows[0] if k not in ('docs', 'bytes')]
    return {
        'docs_per_page': rows[0]['docs'],
        'bytes_per_page': rows[0]['bytes'],
        **{k: round(float(np.median([r[k] for r in rows[1:] or rows])), 4) for k in keys},
    }


async def fetch_grid(client: Any, grid: str, reps: int) -> list[dict[str, Any]]:
    """Time the parallel fetch alone for each ``slices:page_size`` pair of ``grid``."""
    from src.services.curation.clustering import residual_fetch

    rows = []
    for cell in grid.split(','):
        n_slices, page_size = (int(x) for x in cell.split(':'))
        for rep in range(reps):
            t0 = time.perf_counter()
            ids, _ = await residual_fetch.fetch_residual_embeddings_parallel(
                client, n_slices=n_slices, page_size=page_size
            )
            rows.append(
                {
                    'kind': 'fetch_grid',
                    'n_slices': n_slices,
                    'page_size': page_size,
                    'rep': rep,
                    'n': len(ids),
                    'wall_s': round(time.perf_counter() - t0, 3),
                    'loadavg': os.getloadavg()[0],
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    return rows


async def write_grid(client: Any, grid: str, reps: int) -> list[dict[str, Any]]:
    """Time the write-back alone of one fixed labelling for each ``chunk:concurrency`` cell.

    The labels are arbitrary (the cluster-id offset plus the row number modulo 512): the
    cost of a guarded update does not depend on the value written.
    """
    from src.services.curation.clustering import residual_fetch
    from src.services.curation.clustering.assignment_writer import AssignmentWriter, batches

    ids, _ = await residual_fetch.fetch_residual_embeddings_parallel(client)
    rows_all = [(doc_id, 10_000 + i % 512, (i % 100) / 100.0) for i, doc_id in enumerate(ids)]
    out = []
    for cell in grid.split(','):
        chunk, conc = (int(x) for x in cell.split(':'))
        for rep in range(reps):
            t0 = time.perf_counter()
            async with AssignmentWriter(client, 'bench', concurrency=conc) as writer:
                for batch in batches(rows_all, chunk):
                    await writer.submit(batch)
            written = time.perf_counter() - t0
            await client.indices.refresh(index=items_index())
            out.append(
                {
                    'kind': 'write_grid',
                    'chunk': chunk,
                    'concurrency': conc,
                    'rep': rep,
                    'n': len(rows_all),
                    'write_s': round(written, 3),
                    'docs_per_s': round(len(rows_all) / written, 1),
                    'wall_s': round(time.perf_counter() - t0, 3),
                    'loadavg': os.getloadavg()[0],
                }
            )
            print(json.dumps(out[-1]), flush=True)
    return out


async def one_size(args: argparse.Namespace, size: int, out: Path) -> None:
    vectors, truth = load_vectors(args.source, args.data, size)
    # A project of its own per run: its indexes are created through the real mapping and
    # dropped with the run, so no run sees another's documents or cluster state.
    slug = f'{args.project_prefix}-{size}-{uuid.uuid4().hex[:6]}'
    client = make_script_opensearch([args.opensearch_url], timeout=600)
    await ensure_project(client, slug)
    await abind_script_project(slug, opensearch_url=args.opensearch_url)
    rows: list[dict[str, Any]] = []
    load_s = await load_items(client, vectors, truth)
    rows.append({'kind': 'load', 'size': size, 'wall_s': round(load_s, 2)})
    rows.append({'kind': 'dump', 'size': size, 'sha256': fingerprint(vectors)})
    if args.fetch_profile:
        rows.append(
            {
                'kind': 'fetch_profile',
                'size': size,
                **fetch_profile(args.opensearch_url, items_index()),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    if args.write_grid:
        rows.extend(
            {'size': size, **r} for r in await write_grid(client, args.write_grid, args.reps)
        )
    if args.fetch_grid:
        rows.extend(
            {'size': size, **r} for r in await fetch_grid(client, args.fetch_grid, args.reps)
        )
    for method in [m for m in args.methods.split(',') if m]:
        for rep in range(args.reps):
            rows.append(await one_run(client, method, size, rep))
            print(json.dumps(rows[-1], default=float), flush=True)
        if args.geometry:
            rows.append(await geometry_run(client, method, size))
            print(json.dumps(rows[-1], default=float), flush=True)
        if args.digest:
            rows.append(
                {'kind': 'digest', 'method': method, 'size': size, **await label_digest(client)}
            )
            print(json.dumps(rows[-1]), flush=True)
    if args.assign_only:
        for rep in range(args.reps):
            rows.append(await assign_run(client, size, rep))
            print(json.dumps(rows[-1], default=float), flush=True)
        # Again with the assignments in place: the pass a scheduled refresh runs.
        rows.append(await assign_run(client, size, 0, reset=False))
        print(json.dumps(rows[-1], default=float), flush=True)
    with (out / f'{args.label}_runs.jsonl').open('a') as fh:
        fh.writelines(json.dumps({**r, 'label': args.label}, default=float) + '\n' for r in rows)
    if not args.keep:
        await client.indices.delete(index=items_index(), ignore=[404])
    await client.close()


def summarize(path: Path) -> str:
    """Median (min-max) table of the stage seconds per (method, size)."""
    runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = ('wall_s', 'fetch_s', 'fit_s', 'distance_s', 'write_s', 'refresh_s', 'tail_s')
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in runs:
        if r.get('kind') == 'full':
            groups.setdefault((r['method'], r['size']), []).append(r)
    lines = ['| method | n | reps | ' + ' | '.join(keys) + ' |', '|' + '---|' * (3 + len(keys))]
    for (method, size), g in sorted(groups.items()):
        cells = []
        for k in keys:
            s = spread([r[k] for r in g])
            cells.append(f'{s["median"]:.2f} ({s["min"]:.2f}-{s["max"]:.2f})')
        lines.append(f'| {method} | {size} | {len(g)} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--opensearch-url', required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--source', choices=['vecdb', 'parity'], default='vecdb')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', default='run', help='tag for the output file (before, after)')
    ap.add_argument('--sizes', default='2000,10000,100000')
    ap.add_argument('--methods', default='ivf,ahc,hdbscan')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--project-prefix', default='clusplumb')
    ap.add_argument('--fetch-profile', action='store_true')
    ap.add_argument('--fetch-grid', help='time the fetch alone for slices:page_size,... cells')
    ap.add_argument(
        '--write-grid', help='time the write-back alone for chunk:concurrency,... cells'
    )
    ap.add_argument('--geometry', action='store_true', help='also time the cluster_geometry stage')
    ap.add_argument('--assign-only', action='store_true')
    ap.add_argument('--digest', action='store_true', help='hash the written labels per method')
    ap.add_argument('--keep', action='store_true', help='keep the loaded items index')
    ap.add_argument('--summarize', type=Path, help='print the table of a runs file and exit')
    args = ap.parse_args()
    if args.summarize:
        print(summarize(args.summarize))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for size in (int(s) for s in args.sizes.split(',')):
        asyncio.run(one_size(args, size, args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())

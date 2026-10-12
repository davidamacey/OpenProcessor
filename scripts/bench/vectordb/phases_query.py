"""Query phases: recall/latency sweep, operating points, concurrency, deep k, in-set queries."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.vectordb.vdb_lib import (
    FilterSpec,
    latency_summary,
    median_range,
    recall_at_k,
    short_rate,
)


if TYPE_CHECKING:
    from scripts.bench.vectordb.adapters.base import Adapter
    from scripts.bench.vectordb.dataset import Dataset


RECALL_TARGET = 0.95
SWEEP_QUERIES = 120
DETAIL_QUERIES = 300
QPS_FILTERS = ('none', 'review_default', 'project_cluster')
DEEP_FILTERS = ('none', 'review_default')
WARMUP = 200


def timed_search(
    adapter: Adapter, queries: np.ndarray, k: int, flt: FilterSpec, params: dict[str, Any]
) -> tuple[list[list[int]], list[float]]:
    found: list[list[int]] = []
    ms: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        found.append(adapter.search(q, k, flt, params))
        ms.append((time.perf_counter() - t0) * 1000.0)
    return found, ms


def measure(
    adapter: Adapter,
    ds: Dataset,
    qset: str,
    flt: FilterSpec,
    k: int,
    params: dict[str, Any],
    n: int,
) -> dict[str, Any]:
    """One pass of ``n`` queries at concurrency 1: recall@k against the exact answer, latency."""
    queries = (ds.held_queries if qset == 'held' else ds.in_queries)[:n]
    truth = ds.truth(qset, flt)[:n]
    found, ms = timed_search(adapter, queries, k, flt, params)
    return {
        f'recall@{k}': recall_at_k(truth, found, k),
        'short_rate': short_rate(found, truth, k),
        **latency_summary(ms),
    }


def qps(
    adapter: Adapter,
    queries: np.ndarray,
    k: int,
    flt: FilterSpec,
    params: dict[str, Any],
    concurrency: int,
    seconds: float,
) -> dict[str, float]:
    """Closed-loop throughput: ``concurrency`` threads issue queries back to back for ``seconds``."""
    deadline = time.perf_counter() + seconds
    lats: list[list[float]] = [[] for _ in range(concurrency)]

    def worker(idx: int) -> None:
        pos = idx * 7
        while time.perf_counter() < deadline:
            q = queries[pos % len(queries)]
            t0 = time.perf_counter()
            adapter.search(q, k, flt, params)
            lats[idx].append((time.perf_counter() - t0) * 1000.0)
            pos += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in [pool.submit(worker, i) for i in range(concurrency)]:
            fut.result()
    elapsed = time.perf_counter() - t0
    flat = [x for sub in lats for x in sub]
    return {
        'qps': len(flat) / elapsed,
        'p95_ms': latency_summary(flat)['p95_ms'],
        'queries': len(flat),
    }


def sweep(adapter: Adapter, ds: Dataset) -> list[dict[str, Any]]:
    """Recall@10 and latency of every grid setting on every filter (300 held-out queries)."""
    rows = []
    for flt in ds.filters:
        for params in adapter.search_grid():
            row = measure(adapter, ds, 'held', flt, 10, params, SWEEP_QUERIES)
            rows.append({'filter': flt.name, 'params': params, **row})
    return rows


def operating_points(adapter: Adapter, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per filter, the cheapest grid setting with recall@10 >= 0.95, else the most expensive."""
    grid = adapter.search_grid()
    points: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(r['filter'] for r in rows):
        mine = [r for r in rows if r['filter'] == name]
        ok = [r for r in mine if r['recall@10'] >= RECALL_TARGET]
        pick = ok[0] if ok else mine[-1]
        points[name] = {
            'params': pick['params'],
            'reached': bool(ok),
            'grid_index': grid.index(pick['params']),
        }
    return points


def detail(
    adapter: Adapter, ds: Dataset, points: dict[str, dict[str, Any]], reps: int
) -> list[dict[str, Any]]:
    """The full-size measurements at each filter's operating point."""
    out = []
    for flt in ds.filters:
        params = points[flt.name]['params']
        k10_reps = reps if flt.name in QPS_FILTERS else 1
        k10 = [
            measure(adapter, ds, 'held', flt, 10, params, DETAIL_QUERIES) for _ in range(k10_reps)
        ]
        k100_params, k100 = deep_recall(adapter, ds, flt, params)
        row: dict[str, Any] = {
            'filter': flt.name,
            'params': params,
            'reached_target': points[flt.name]['reached'],
            'selectivity': float(np.mean(_mask(ds, flt))),
            'recall@10': k10[0]['recall@10'],
            'short_rate@10': k10[0]['short_rate'],
            'p50_ms': median_range([r['p50_ms'] for r in k10]),
            'p95_ms': median_range([r['p95_ms'] for r in k10]),
            'p99_ms': median_range([r['p99_ms'] for r in k10]),
            'recall@100': k100['recall@100'],
            'short_rate@100': k100['short_rate'],
            'k100_params': k100_params,
            'k100_p95_ms': k100['p95_ms'],
        }
        if flt.name in QPS_FILTERS:
            for conc in (8, 32):
                runs = [
                    qps(adapter, ds.held_queries, 10, flt, params, conc, 3.0) for _ in range(reps)
                ]
                row[f'qps_c{conc}'] = median_range([r['qps'] for r in runs])
                row[f'p95_ms_c{conc}'] = median_range([r['p95_ms'] for r in runs])
        if flt.name in DEEP_FILTERS:
            deep = measure(adapter, ds, 'held', flt, min(2000, ds.size), params, 100)
            row['k2000_p95_ms'] = deep['p95_ms']
            row['k2000_short_rate'] = deep['short_rate']
            row['in_set@10'] = measure(adapter, ds, 'in', flt, 10, params, DETAIL_QUERIES)
        out.append(row)
    return out


def deep_recall(
    adapter: Adapter, ds: Dataset, flt: FilterSpec, params: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """k=100 at the operating point, stepping up the grid until recall@100 reaches the target."""
    grid = adapter.search_grid()
    for p in grid[grid.index(params) :]:
        res = measure(adapter, ds, 'held', flt, 100, p, DETAIL_QUERIES)
        if res['recall@100'] >= RECALL_TARGET:
            return p, res
    return grid[-1], res


def _mask(ds: Dataset, flt: FilterSpec) -> np.ndarray:
    from scripts.bench.vectordb.vdb_lib import filter_mask

    return filter_mask(flt, ds.meta)


def query_phase(adapter: Adapter, ds: Dataset, reps: int) -> dict[str, Any]:
    first = adapter.search_grid()[0]
    timed_search(adapter, ds.held_queries[:WARMUP], 10, ds.filters[0], first)
    rows = sweep(adapter, ds)
    points = operating_points(adapter, rows)
    return {'sweep': rows, 'operating_points': points, 'detail': detail(adapter, ds, points, reps)}


class BackgroundQueries:
    """Queries issued from a thread while another phase runs, for recall/latency during a write."""

    def __init__(
        self, adapter: Adapter, ds: Dataset, flt: FilterSpec, params: dict[str, Any]
    ) -> None:
        self.adapter, self.ds, self.flt, self.params = adapter, ds, flt, params
        self.found: list[list[int]] = []
        self.ms: list[float] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        pos = 0
        n = len(self.ds.held_queries)
        while not self.stop.is_set():
            q = self.ds.held_queries[pos % n]
            t0 = time.perf_counter()
            self.found.append(self.adapter.search(q, 10, self.flt, self.params))
            self.ms.append((time.perf_counter() - t0) * 1000.0)
            pos += 1

    def __enter__(self) -> BackgroundQueries:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.thread.join()

    def summary(self) -> dict[str, Any]:
        n = len(self.found)
        if not n:
            return {'queries': 0}
        truth = self.ds.truth('held', self.flt)
        idx = [i % len(truth) for i in range(n)]
        return {
            'queries': n,
            'recall@10': recall_at_k(truth[idx], self.found, 10),
            **latency_summary(self.ms),
        }

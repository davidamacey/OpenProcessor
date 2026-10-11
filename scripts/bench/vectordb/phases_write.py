"""Write-side phases: load, metadata rewrite, streaming insert, delete, resource footprint."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.vectordb import hostinfo
from scripts.bench.vectordb.phases_query import BackgroundQueries, measure, qps
from scripts.bench.vectordb.vdb_lib import CLUSTER_COUNT, FilterSpec, bytes_per_vector


if TYPE_CHECKING:
    from scripts.bench.vectordb.adapters.base import Adapter
    from scripts.bench.vectordb.dataset import Dataset


STREAM_RATE = 200  # vectors per second offered (plan section 6)
STREAM_SECONDS = 20.0
REWRITE_SIZES = (12_800,)


def load_phase(adapter: Adapter, ds: Dataset, batch: int, threads: int) -> dict[str, Any]:
    """Create, load and finish. Times are wall clock; ``searchable_s`` is load plus finish."""
    mem0 = hostinfo.cgroup_memory(adapter.container) if adapter.container else None
    before_anon = mem0['anon'] if mem0 else None
    before_shmem = mem0['shmem'] if mem0 else None
    before_du = (
        hostinfo.du_bytes(adapter.container, adapter.data_path)
        if adapter.container and adapter.data_path
        else None
    )
    adapter.create(ds.vectors.shape[1])
    adapter.prepare(ds.vectors, ds.meta, batch)
    t0 = time.perf_counter()
    adapter.load(threads)
    t1 = time.perf_counter()
    adapter.finish_load()
    t2 = time.perf_counter()
    n = len(ds.vectors)
    return {
        'cgroup_anon_before_bytes': before_anon,
        'cgroup_shmem_before_bytes': before_shmem,
        'data_dir_before_bytes': before_du,
        'threads': threads,
        'batch': batch,
        'load_s': t1 - t0,
        'finish_s': t2 - t1,
        'searchable_s': t2 - t0,
        'vectors_per_s': n / (t1 - t0),
        'vectors_per_s_to_searchable': n / (t2 - t0),
    }


def footprint(adapter: Adapter, n: int) -> dict[str, Any]:
    """Disk bytes per vector after settle, memory of the container, idle CPU."""
    out: dict[str, Any] = {}
    reported = adapter.bytes_on_disk()
    if reported >= 0:
        out['engine_reported_bytes'] = reported
        out['engine_reported_bytes_per_vector'] = bytes_per_vector(reported, n)
    if adapter.container and adapter.data_path:
        du = hostinfo.du_bytes(adapter.container, adapter.data_path)
        out['data_dir_bytes'] = du
        out['data_dir_bytes_per_vector'] = bytes_per_vector(du, n)
        mem = hostinfo.cgroup_memory(adapter.container)
        out['cgroup_anon_bytes'] = mem['anon']
        out['cgroup_file_cache_bytes'] = mem['file']
        out['cgroup_shmem_bytes'] = mem['shmem']
        out['cgroup_current_bytes'] = mem['current']
        out['cgroup_anon_bytes_per_vector'] = bytes_per_vector(mem['anon'], n)
        c0 = hostinfo.cpu_seconds(adapter.container)
        time.sleep(10)
        out['idle_cpu_cores'] = (hostinfo.cpu_seconds(adapter.container) - c0) / 10
    out['engine_stats'] = adapter.engine_stats()
    return out


def rewrite_phase(
    adapter: Adapter, ds: Dataset, op: dict[str, Any], threads: int
) -> list[dict[str, Any]]:
    """Rewrite ``cluster_id`` (the cluster-pass write-back) while review queries keep running."""
    review = next(f for f in ds.filters if f.name == 'review_default')
    params = op[review.name]['params']
    sizes = [s for s in (*REWRITE_SIZES, ds.size) if s <= ds.size]
    results = []
    for size in dict.fromkeys(sizes):
        ids = np.arange(size)
        new = (ds.meta['cluster_id'][:size] + 1) % CLUSTER_COUNT
        with BackgroundQueries(adapter, ds, review, params) as bg:
            t0 = time.perf_counter()
            adapter.update_field(ids, 'cluster_id', new, threads)
            seconds = time.perf_counter() - t0
        after = measure(adapter, ds, 'held', review, 10, params, 300)
        results.append(
            {
                'rows': size,
                'seconds': seconds,
                'rows_per_s': size / seconds,
                'queries_during': bg.summary(),
                'after': after,
            }
        )
    return results


def stream_phase(adapter: Adapter, ds: Dataset, op: dict[str, Any]) -> dict[str, Any]:
    """Query p95 at 4 clients idle, then while 200 vectors/s are inserted (plan section 6)."""
    review = next(f for f in ds.filters if f.name == 'review_default')
    params = op[review.name]['params']
    base = qps(adapter, ds.held_queries, 10, review, params, 4, STREAM_SECONDS)
    pool = len(ds.stream_vectors)
    inserted = [0]
    stop = threading.Event()
    lock = threading.Lock()
    chunk = 20
    started = [0.0]

    def writer() -> None:
        # Four writers share one schedule of ``chunk`` vectors every chunk / STREAM_RATE seconds.
        while not stop.is_set():
            with lock:
                pos = inserted[0]
                if pos + chunk > pool:
                    return
                inserted[0] = pos + chunk
            wait = started[0] + pos / STREAM_RATE - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            sub = slice(pos, pos + chunk)
            meta = {k: v[sub] for k, v in ds.stream_meta.items()}
            ids = np.arange(ds.size + pos, ds.size + pos + chunk)
            adapter.insert_rows(ds.stream_vectors[sub], meta, ids)
            done[0] += chunk

    done = [0]
    threads = [threading.Thread(target=writer, daemon=True) for _ in range(4)]
    t0 = time.perf_counter()
    started[0] = t0
    for th in threads:
        th.start()
    mixed = qps(adapter, ds.held_queries, 10, review, params, 4, STREAM_SECONDS)
    stop.set()
    for th in threads:
        th.join()
    elapsed = time.perf_counter() - t0
    return {
        'offered_per_s': STREAM_RATE,
        'achieved_per_s': done[0] / elapsed,
        'idle': base,
        'mixed': mixed,
        'p95_ratio': mixed['p95_ms'] / base['p95_ms'],
    }


def delete_phase(adapter: Adapter, flt: FilterSpec) -> dict[str, Any]:
    """Delete one project (about 10 percent): time, then bytes before, after and after settle."""
    before_rows = adapter.count()
    before = footprint_bytes(adapter)
    t0 = time.perf_counter()
    adapter.delete_where(flt)
    delete_s = time.perf_counter() - t0
    after = footprint_bytes(adapter)
    t1 = time.perf_counter()
    adapter.settle()
    settle_s = time.perf_counter() - t1
    settled = footprint_bytes(adapter)
    return {
        'rows_before': before_rows,
        'rows_after': adapter.count(),
        'delete_s': delete_s,
        'settle_s': settle_s,
        'bytes_before': before,
        'bytes_after_delete': after,
        'bytes_after_settle': settled,
    }


def footprint_bytes(adapter: Adapter) -> int:
    if adapter.container and adapter.data_path:
        return hostinfo.du_bytes(adapter.container, adapter.data_path)
    return adapter.bytes_on_disk()

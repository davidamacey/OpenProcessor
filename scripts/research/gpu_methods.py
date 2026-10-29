"""GPU methods for the Leiden prototype: GPU kNN, cuGraph Leiden and ECG, cuML HDBSCAN (research).

Imported by ``leiden_refine_proto.py`` for ``gpu_*`` methods; run it with
``CUDA_VISIBLE_DEVICES=<one physical GPU>`` in the GPU environment (cugraph, cuml, cupy, faiss-gpu,
pynvml). cuGraph Leiden optimises modularity (``resolution`` 1.0 is the usual value); it has no
CPM objective and takes no initial partition, so it is a "plain" method, compared against the CPU
modularity Leiden for like-for-like quality.

A watchdog thread polls the GPU's used memory; it records the peak above the starting level and
aborts the process if the run exceeds ``--gpu-limit-mb`` so a shared card is never filled.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np


class GpuWatch:
    def __init__(self, limit_mb: float, device: int = 0) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nv = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(device)
        self.base = self._used()
        self.peak = 0.0
        self.limit = limit_mb
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _used(self) -> float:
        return float(self._nv.nvmlDeviceGetMemoryInfo(self._h).used) / 2**20

    def _loop(self) -> None:
        while not self._stop.wait(0.05):
            used = self._used() - self.base
            self.peak = max(self.peak, used)
            if used > self.limit:
                os.write(2, f'GPU memory {used:.0f} MB over limit {self.limit:.0f}\n'.encode())
                os._exit(3)

    def stop(self) -> float:
        self._stop.set()
        self._t.join()
        return round(self.peak, 1)


def knn_faiss_gpu(vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), vectors.shape[1])
    index.add(vectors)
    sims, idx = index.search(vectors, k + 1)
    return _drop_self(sims, idx, k)


def knn_cuvs(vectors: np.ndarray, k: int, batch: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp
    from cuvs.neighbors import brute_force

    data = cp.asarray(vectors)
    index = brute_force.build(data, metric='inner_product')
    sims = np.empty((len(vectors), k + 1), dtype=np.float32)
    idx = np.empty((len(vectors), k + 1), dtype=np.int64)
    for s in range(0, len(vectors), batch):
        d, i = brute_force.search(index, data[s : s + batch], k + 1)
        sims[s : s + batch] = cp.asnumpy(cp.asarray(d))
        idx[s : s + batch] = cp.asnumpy(cp.asarray(i))
    return _drop_self(sims, idx, k)


def _drop_self(sims: np.ndarray, idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(idx)
    is_self = idx == np.arange(n)[:, None]
    is_self[~is_self.any(axis=1), -1] = True
    keep = ~is_self
    return sims[keep].reshape(n, k), idx[keep].reshape(n, k)


def mutual_edges_gpu(sims: np.ndarray, idx: np.ndarray) -> tuple[Any, Any, Any]:
    """Mutual-kNN upper-triangle edge list on the GPU (cupy sparse), weight = cosine."""
    import cupy as cp
    import cupyx.scipy.sparse as csp

    n, k = idx.shape
    w = cp.asarray(np.maximum(sims, 0.0).astype(np.float32).ravel())
    rows = cp.repeat(cp.arange(n, dtype=cp.int32), k)
    directed = csp.csr_matrix((w, (rows, cp.asarray(idx.ravel(), dtype=cp.int32))), shape=(n, n))
    directed.eliminate_zeros()
    mutual = directed.minimum(directed.T)
    mutual.eliminate_zeros()
    upper = csp.triu(mutual, k=1).tocoo()
    return upper.row, upper.col, upper.data


def _to_graph(src: Any, dst: Any, wt: Any) -> Any:
    import cudf
    import cugraph

    df = cudf.DataFrame({'src': src, 'dst': dst, 'weight': wt})
    graph = cugraph.Graph()
    graph.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
    return graph


def _labels_from(df: Any, n: int, vertex: str = 'vertex', part: str = 'partition') -> np.ndarray:
    out = np.full(n, -1, dtype=np.int64)
    v, p = df[vertex].to_numpy(), df[part].to_numpy()
    out[v] = p
    miss = out < 0  # isolated vertices never entered the edge list: their own cluster
    out[miss] = out.max() + 1 + np.arange(int(miss.sum()))
    return out


def run(method: str, vectors: np.ndarray, args: Any, tm: Any, info: dict[str, Any]) -> np.ndarray:
    import cupy as cp

    watch = GpuWatch(getattr(args, 'gpu_limit_mb', 40_000))
    try:
        labels = _run(method, vectors, args, tm, info)
        cp.cuda.Device().synchronize()
    finally:
        info['gpu_peak_mb'] = watch.stop()
    return labels


def _run(method: str, vectors: np.ndarray, args: Any, tm: Any, info: dict[str, Any]) -> np.ndarray:
    import cupy as cp

    n = len(vectors)
    if method == 'gpu_hdbscan':
        from cuml.cluster import HDBSCAN

        info['hdbscan_build'] = 'brute_force' if n <= 100_000 else 'nn_descent'

        def fit() -> np.ndarray:
            model = HDBSCAN(
                min_cluster_size=5,
                min_samples=5,
                cluster_selection_method='leaf',
                build_algo=info['hdbscan_build'],
            )
            return np.asarray(cp.asnumpy(model.fit_predict(cp.asarray(vectors)))).astype(np.int64)

        out: np.ndarray = tm.run('cluster_s', fit)
        return out
    backend = os.environ.get('OP_KNN_BACKEND', 'faiss')
    info['knn_backend'] = backend
    knn = knn_faiss_gpu if backend == 'faiss' else knn_cuvs
    sims, idx = tm.run('knn_s', knn, vectors, args.k)
    src, dst, wt = tm.run('graph_s', mutual_edges_gpu, sims, idx)
    info['edges'] = len(src)
    graph = _to_graph(src, dst, wt)
    import cugraph

    t = time.perf_counter()
    if method == 'gpu_leiden':
        parts, _ = cugraph.leiden(
            graph, max_iter=100, resolution=args.gamma, random_state=args.seed
        )
    elif method == 'gpu_ecg':
        parts = cugraph.ecg(
            graph, ensemble_size=args.consensus_runs, resolution=args.gamma, random_state=args.seed
        )
    else:
        raise SystemExit(f'unknown GPU method {method}')
    tm.stages['cluster_s'] = round(time.perf_counter() - t, 3)
    return _labels_from(parts, n)

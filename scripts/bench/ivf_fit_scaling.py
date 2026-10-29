"""Time the production IVF k-means (train) and full-pool assign at several training sizes.

Isolates the fit stage of the cluster pass from OpenSearch: the same ``IVFCentroidStore.train``
call (spherical FAISS k-means, 512 centroids, 50 iterations, 5 restarts) on nested prefixes of
a ``vecdb`` npz, then the flat nearest-centroid assign of the whole prefix. The achieved
GFLOP/s of the train (``2 * n * k * d * niter * nredo`` flops) shows whether the cost is
the arithmetic or the machine.

    ivf_fit_scaling.py --data vectors.npz --sizes 10000,25000,50000 --out fit_scaling.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.cluster_plumbing_bench import load_vectors
from src.services.curation.clustering.methods.ivf import (
    IVF_DEFAULT_N_CLUSTERS,
    IVF_DEFAULT_NITER,
    IVF_DEFAULT_NREDO,
)
from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--sizes', default='10000,25000,50000')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(',')]
    pool, _ = load_vectors('vecdb', args.data, max(sizes))
    rows = []
    scratch = tempfile.mkdtemp(prefix='ivf_fit_scaling_')
    for n in sizes:
        x = np.ascontiguousarray(pool[:n])
        for rep in range(args.reps):
            store = IVFCentroidStore(Path(scratch))
            t0 = time.perf_counter()
            centroids, _ = store.train(
                x,
                n_clusters=IVF_DEFAULT_N_CLUSTERS,
                niter=IVF_DEFAULT_NITER,
                nredo=IVF_DEFAULT_NREDO,
                use_gpu=False,
            )
            train_s = time.perf_counter() - t0
            import faiss

            index = faiss.IndexFlatL2(x.shape[1])
            index.add(centroids)
            store._index = index
            t0 = time.perf_counter()
            store.assign_batch_with_distances(x)
            assign_s = time.perf_counter() - t0
            flops = (
                2 * n * IVF_DEFAULT_N_CLUSTERS * x.shape[1] * IVF_DEFAULT_NITER * IVF_DEFAULT_NREDO
            )
            rows.append(
                {
                    'n': n,
                    'rep': rep,
                    'train_s': round(train_s, 2),
                    'assign_s': round(assign_s, 3),
                    'train_gflops': round(flops / train_s / 1e9, 1),
                    'loadavg': os.getloadavg()[0],
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    args.out.write_text(json.dumps(rows, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

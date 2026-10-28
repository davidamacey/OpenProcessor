"""Seeded synthetic embeddings with planted structure, for cluster scale-up beyond the real set.

Each vector is a mixture-component centroid plus isotropic gaussian noise, L2-normalised. The
centroids are sampled from the real vectors of ``--base`` (a ``vecdb`` npz), so the component
geometry is that of real PE-Core crop embeddings; the component weights are log-normal so the
classes are unbalanced like real crop sets. The output is the same ``vecdb`` npz
(``vectors`` float32, ``class_name`` = ``c<component>``) that ``cluster_ladder.py`` reads, and
sizes are nested prefixes (rows are drawn i.i.d., so a prefix is a random subsample).

    gen_synthetic_embeddings.py --base setP.npz --out synth.npz --n 1000000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def generate(
    centroids: np.ndarray, n: int, *, sigma: float, seed: int, chunk: int = 50_000
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k, dim = centroids.shape
    raw = np.asarray(rng.lognormal(0.0, 1.0, size=k))
    weights = raw / raw.sum()
    comp = rng.choice(k, size=n, p=weights)
    vectors = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        noise = rng.standard_normal((stop - start, dim), dtype=np.float32) * (sigma / np.sqrt(dim))
        block = centroids[comp[start:stop]] + noise
        vectors[start:stop] = block / np.linalg.norm(block, axis=1, keepdims=True)
    return vectors, np.char.add('c', comp.astype(str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        '--base', type=Path, required=True, help='real vecdb npz to draw centroids from'
    )
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=1_000_000)
    ap.add_argument('--components', type=int, default=1500)
    ap.add_argument('--sigma', type=float, default=0.6, help='noise norm; centroid cosine ~0.86')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    base = np.load(args.base)['vectors']
    pick = np.random.default_rng(args.seed).choice(len(base), size=args.components, replace=False)
    vectors, classes = generate(base[pick], args.n, sigma=args.sigma, seed=args.seed)
    np.savez(args.out, vectors=vectors, class_name=classes)
    print(f'wrote {args.n} x {vectors.shape[1]} in {args.components} components -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

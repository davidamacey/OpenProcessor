"""Recall@k of the FAISS IVF kNN used above ``--exact-below`` rows, against exact search on a sample.

knn_recall.py --vectors v.npz --k 30 --queries 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ruff: noqa: E402
import graph_lib as gl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--vectors', type=Path, required=True)
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--queries', type=int, default=2000)
    args = ap.parse_args()
    v = np.load(args.vectors)['vectors'].astype(np.float32)
    _, approx = gl.knn_cpu(v, args.k, exact_below=0)
    pick = np.random.default_rng(0).choice(len(v), size=args.queries, replace=False)
    exact = np.argsort(-(v[pick] @ v.T), axis=1)[:, 1 : args.k + 1]
    hit = np.mean([len(set(a) & set(b)) / args.k for a, b in zip(approx[pick], exact, strict=True)])
    print(f'recall@{args.k} of IVF (nprobe 16) vs exact on {args.queries} queries: {hit:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

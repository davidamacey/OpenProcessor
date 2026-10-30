"""Seeded synthetic embeddings with a planted partition, bridges, mixed items and noise.

    planted.py --out synth_20k.npz --n 20000 --seed 7

Classes come in groups of five siblings whose centres share a group centre (sibling centre cosine
about 0.73); every third group also holds a "twin" class whose centre is almost on top of its
sibling (cosine about 0.95), which no geometry can separate and so measures false merges honestly.
Items are class centre plus isotropic noise. On top of the clean items:

* bridges: points between two class centres, labelled with the nearer class;
* mixed items: drawn in a sibling's region but labelled with another class of the group;
* noise: random unit vectors, each its own singleton class (they belong to no cluster).

Class sizes are log-normal. The npz holds ``vectors`` (float32, L2-normalised), ``item_id`` and
``label``. Public synthetic data only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return out


def planted(
    n: int,
    *,
    dim: int = 128,
    seed: int = 7,
    sigma: float = 0.5,
    mean_class: int = 150,
    bridge: float = 0.03,
    mixed: float = 0.02,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_noise, n_bridge, n_mixed = (int(n * f) for f in (noise, bridge, mixed))
    n_clean = n - n_noise - n_bridge - n_mixed
    n_class = max(10, n_clean // mean_class)
    groups = -(-n_class // 5)
    group_c = _unit(rng.standard_normal((groups, dim)))
    centres = _unit(
        group_c[np.arange(n_class) // 5] + 0.6 * _unit(rng.standard_normal((n_class, dim)))
    )
    for g in range(0, groups, 3):  # make class 5g+1 a twin of 5g
        if 5 * g + 1 < n_class:
            centres[5 * g + 1] = _unit(centres[5 * g] + 0.33 * _unit(rng.standard_normal(dim)))
    w = rng.lognormal(0.0, 0.7, n_class)
    comp = rng.choice(n_class, size=n_clean, p=w / w.sum())
    clean = _unit(centres[comp] + sigma / np.sqrt(dim) * rng.standard_normal((n_clean, dim)))
    labels = [np.char.add('c', comp.astype(str))]
    parts = [clean]

    a = rng.integers(0, n_class, n_bridge)
    sib = np.where(rng.random(n_bridge) < 0.7, (a // 5) * 5 + rng.integers(0, 5, n_bridge), -1)
    b = np.where((sib >= 0) & (sib < n_class), sib, rng.integers(0, n_class, n_bridge))
    t = rng.uniform(0.3, 0.7, n_bridge)[:, None]
    br = _unit(
        t * centres[a]
        + (1 - t) * centres[b]
        + 0.3 / np.sqrt(dim) * rng.standard_normal((n_bridge, dim))
    )
    nearer = np.where(t[:, 0] >= 0.5, a, b)
    parts.append(br)
    labels.append(np.char.add('c', nearer.astype(str)))

    home = rng.choice(n_class, size=n_mixed, p=w / w.sum())
    wrong = np.minimum((home // 5) * 5 + (home % 5 + rng.integers(1, 5, n_mixed)) % 5, n_class - 1)
    mx = _unit(centres[home] + sigma / np.sqrt(dim) * rng.standard_normal((n_mixed, dim)))
    parts.append(mx)
    labels.append(np.char.add('c', wrong.astype(str)))

    parts.append(_unit(rng.standard_normal((n_noise, dim))))
    labels.append(np.char.add('noise', np.arange(n_noise).astype(str)))

    vectors = np.concatenate(parts).astype(np.float32)
    label = np.concatenate(labels)
    order = rng.permutation(n)
    return vectors[order], label[order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=20_000)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--sigma', type=float, default=0.5, help='noise norm around a class centre')
    args = ap.parse_args()
    vectors, label = planted(args.n, dim=args.dim, seed=args.seed, sigma=args.sigma)
    ids = np.char.add('i', np.arange(args.n).astype(str))
    np.savez(args.out, vectors=vectors, item_id=ids, label=label)
    print(f'wrote {args.n} x {args.dim}, {len(np.unique(label))} label values -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

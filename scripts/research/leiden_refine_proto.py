"""Offline prototype: global kNN graph + Leiden refinement of an existing partition (research).

    leiden_refine_proto.py run --vectors v.npz --method leiden_chameleon --gamma 0.02 \\
        --partition current.jsonl --out out/leiden_chameleon.jsonl [--seed 0]

Reads L2-normalised vectors (npz with ``vectors`` and ``item_id``, or JSONL rows
``{"item_id", "vector"}``), optionally the current partition (JSONL ``item_id``, ``cluster_id``),
builds one global cosine kNN graph, runs the chosen method and writes the candidate partition as
JSONL for ``scripts/bench/cluster_eval.py score``, plus ``<out>.stats.json`` with wall time per
stage and the process's peak RSS. Nothing is written to a project; one method per process so the
peak RSS belongs to that method alone.

Methods (CPU): ivf, ivf_ahc, hdbscan_leaf, finch0, finch1, leiden_plain, leiden_init,
leiden_chameleon, consensus, leiden_mod. GPU methods live in ``gpu_methods.py`` (gpu_leiden,
gpu_ecg, gpu_hdbscan) and need the GPU environment.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar('T')
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ruff: noqa: E402
import baselines
import graph_lib as gl


GRAPH_METHODS = {'leiden_plain', 'leiden_init', 'leiden_chameleon', 'consensus', 'leiden_mod'}


def load_vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == '.npz':
        z = np.load(path)
        return z['vectors'].astype(np.float32), z['item_id'].astype(str)
    ids, vecs = [], []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                ids.append(str(row['item_id']))
                vecs.append(row['vector'])
    v = np.asarray(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True), np.array(ids)


def load_partition(path: Path, ids: np.ndarray) -> np.ndarray:
    by_item: dict[str, int] = {}
    code: dict[str, int] = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                cid = row.get('cluster_id')
                if cid is not None:
                    by_item[str(row['item_id'])] = code.setdefault(str(cid), len(code))
    return np.array([by_item.get(i, gl.UNASSIGNED) for i in ids.tolist()], dtype=np.int64)


def write_partition(path: Path, ids: np.ndarray, labels: np.ndarray) -> None:
    with path.open('w') as fh:
        for i, c in zip(ids.tolist(), labels.tolist(), strict=True):
            fh.write(json.dumps({'item_id': i, 'cluster_id': None if c < 0 else c}) + '\n')


class Timer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def run(self, name: str, fn: Callable[..., T], *a: Any, **kw: Any) -> T:
        t = time.perf_counter()
        out = fn(*a, **kw)
        self.stages[name] = round(time.perf_counter() - t, 3)
        return out


def build_graph(vectors: np.ndarray, args: argparse.Namespace, tm: Timer) -> Any:
    sims, idx = tm.run('knn_s', gl.knn_cpu, vectors, args.k, threads=args.threads)
    return tm.run(
        'graph_s',
        gl.mutual_graph,
        sims,
        idx,
        min_sim=args.min_sim,
        adaptive_beta=args.adaptive_beta,
    )


def run_graph_method(
    vectors: np.ndarray, ids: np.ndarray, args: argparse.Namespace, tm: Timer, info: dict[str, Any]
) -> np.ndarray:
    graph = build_graph(vectors, args, tm)
    deg = np.diff(graph.indptr)
    info['edges'] = int(graph.nnz // 2)
    info['isolated_nodes'] = int((deg == 0).sum())
    info['mean_degree'] = round(float(deg.mean()), 2)
    if args.method == 'leiden_mod':
        return tm.run('cluster_s', gl.leiden_modularity, graph, args.gamma, seed=args.seed)
    if args.method == 'consensus':
        return tm.run(
            'cluster_s',
            gl.consensus_leiden,
            graph,
            args.gamma,
            runs=args.consensus_runs,
            seed=args.seed,
        )
    if args.method == 'leiden_plain':
        return tm.run('cluster_s', gl.leiden_cpm, graph, args.gamma, seed=args.seed)
    if not args.partition:
        raise SystemExit(f'{args.method} needs --partition (the current partition to start from)')
    base = load_partition(Path(args.partition), ids)
    moved = tm.run('cluster_s', gl.leiden_cpm, graph, args.gamma, init=base, seed=args.seed)
    if args.method == 'leiden_init':
        return np.asarray(moved)
    atoms = gl.codes(base, moved)
    info['atoms'] = int(atoms.max()) + 1
    return tm.run(
        'chameleon_s', gl.chameleon_merge, graph, atoms, ri_min=args.ri_min, rc_min=args.rc_min
    )


def cluster(
    vectors: np.ndarray, ids: np.ndarray, args: argparse.Namespace, tm: Timer, info: dict[str, Any]
) -> np.ndarray:
    m = args.method
    if m in {'ivf', 'ivf_ahc'}:
        bucket = tm.run('ivf_s', baselines.ivf_buckets, vectors, seed=args.seed)
        return bucket if m == 'ivf' else tm.run('ahc_s', baselines.ahc_refine, vectors, bucket)
    if m == 'hdbscan_leaf':
        return tm.run('cluster_s', baselines.hdbscan_leaf, vectors, threads=args.threads)
    if m in {'finch0', 'finch1'}:
        levels = tm.run('cluster_s', baselines.finch_partitions, vectors)
        info['finch_levels'] = [int(lv.max()) + 1 for lv in levels]
        return np.asarray(levels[int(m[-1])])
    if m in GRAPH_METHODS:
        return run_graph_method(vectors, ids, args, tm, info)
    if m.startswith('gpu_'):
        import gpu_methods

        return np.asarray(gpu_methods.run(m, vectors, args, tm, info))
    raise SystemExit(f'unknown method {m}')


def run(args: argparse.Namespace) -> dict[str, Any]:
    tm = Timer()
    vectors, ids = load_vectors(Path(args.vectors))
    info: dict[str, Any] = {}
    started = time.perf_counter()
    labels = cluster(vectors, ids, args, tm, info)
    wall = time.perf_counter() - started
    if (
        args.method not in {'ivf', 'ivf_ahc', 'hdbscan_leaf', 'gpu_hdbscan'}
        and not args.native_assign
    ):
        labels = gl.apply_min_size(labels, args.min_size)
    stats = {
        'method': args.method,
        'seed': args.seed,
        'items': len(ids),
        'gamma': args.gamma,
        'k': args.k,
        'ri_min': args.ri_min,
        'rc_min': args.rc_min,
        'min_size': args.min_size,
        'clusters': int(labels.max()) + 1 if (labels >= 0).any() else 0,
        'unassigned': int((labels < 0).sum()),
        'wall_s': round(wall, 3),
        'stages': tm.stages,
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        **info,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_partition(out, ids, labels)
    out.with_suffix('.stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='run one method and write its partition')
    r.add_argument('--vectors', required=True)
    r.add_argument('--method', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--partition', help='current partition JSONL (leiden_init, leiden_chameleon)')
    r.add_argument('--k', type=int, default=30)
    r.add_argument('--gamma', type=float, default=0.02, help='CPM resolution (modularity: 1.0)')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--min-sim', type=float, default=0.0)
    r.add_argument('--adaptive-beta', type=float)
    r.add_argument('--ri-min', type=float, default=0.3)
    r.add_argument('--rc-min', type=float, default=0.9)
    r.add_argument('--min-size', type=int, default=3, help='smaller clusters become unassigned')
    r.add_argument('--native-assign', action='store_true', help='skip the min-size rule')
    r.add_argument('--consensus-runs', type=int, default=5)
    r.add_argument('--threads', type=int, default=6)
    r.add_argument('--gpu-limit-mb', type=float, default=40_000, help='abort a GPU run above this')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

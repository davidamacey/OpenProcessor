"""Agreement gate for the IVF fit's restart count (``nredo``).

``nredo`` restarts of the k-means fit keep the best objective, so the fit costs ``nredo`` times
one restart. This scores a smaller count against the incumbent (``--incumbent``, 5) on real
vectors, with the production fit (spherical FAISS k-means, 512 centroids, 50 iterations, the
50,000-vector training sample, nearest-centroid assignment of the whole pool):

* agreement of the candidate's partition with the incumbent's: ARI, pair precision and recall of
  the candidate's co-clustered pairs against the incumbent's, and the share of items whose
  cluster is the one the incumbent cluster maps to;
* the noise floor: the same agreement between the incumbent and an incumbent refit with other
  seeds, since k-means partitions at k=512 differ between any two seeds;
* label quality when labels exist: purity (1 - minority item rate) and the false-merge rate
  (``cluster_eval_metrics.pairwise_scores``);
* the fit time of each count (indicative only; the ladder carries the timed numbers).

The gate: purity not lower than the incumbent's by more than 0.5 point and the false-merge rate
not higher by more than 0.5 point, and pair agreement with the incumbent (ARI) not below the
noise floor by more than 0.02. Two k-means fits of one pool with different seeds already differ
by 2 to 3 points of purity and false-merge rate at k=512, so ``judge_mean`` compares the means
over all seeds (candidate against incumbent, same seeds); ``judge`` on one seed is reported
beside it as the spread, not as the verdict.

    ivf_nredo_agreement.py --vectors v.npy --labels labels.jsonl --meta v.meta.jsonl \\
        --candidates 1,2,3 --seeds 7,13 --out agreement.json

``--labels`` rows are ``item_id`` and ``label`` (as ``cluster_eval.py`` reads them), joined to the
vectors through ``crop_id`` of ``--meta``; ``--meta-label-field detector_class_name`` instead uses
that field of each meta row as the label (a proxy when no ground truth exists).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.cluster_eval_metrics import moved_items, pairwise_scores
from scripts.bench.parity_metrics import adjusted_rand_index
from src.services.curation.clustering.methods.ivf import (
    IVF_DEFAULT_N_CLUSTERS,
    IVF_DEFAULT_NITER,
    IVF_MAX_TRAIN_SAMPLE,
)


PRODUCTION_SEED = 42
PURITY_DROP_MAX = 0.005
FALSE_MERGE_RISE_MAX = 0.005
ARI_BELOW_FLOOR_MAX = 0.02


def train_centroids(
    x: np.ndarray, *, n_clusters: int, niter: int, nredo: int, seed: int
) -> np.ndarray:
    """The ``IVFCentroidStore.train`` fit with a seed argument (its seed is fixed at 42)."""
    import faiss

    kmeans = faiss.Kmeans(
        x.shape[1], n_clusters, niter=niter, nredo=nredo, verbose=False, seed=seed, spherical=True
    )
    kmeans.train(x)
    c = np.asarray(kmeans.index.reconstruct_n(0, n_clusters), dtype=np.float32)
    c = c.reshape(n_clusters, x.shape[1])
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    return np.ascontiguousarray(c / np.where(norms == 0, 1.0, norms), dtype=np.float32)


def assign(centroids: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """``(nearest centroid per row, mean cosine to it)``; unit rows and centroids."""
    import faiss

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(centroids)
    sq, nearest = index.search(x, 1)
    return nearest[:, 0].astype(np.int64), float(np.mean(1.0 - sq[:, 0] / 2.0))


def fit_partition(
    x: np.ndarray, *, nredo: int, seed: int = PRODUCTION_SEED
) -> tuple[np.ndarray, float, float]:
    """``(labels, fit seconds, mean cosine to the centroid)``: production sampling, k-means, assignment of the whole pool."""
    n = x.shape[0]
    n_clusters = min(IVF_DEFAULT_N_CLUSTERS, max(1, n // 6))
    train_x = x
    if n > IVF_MAX_TRAIN_SAMPLE:
        idx = np.random.default_rng(42).choice(n, size=IVF_MAX_TRAIN_SAMPLE, replace=False)
        train_x = np.ascontiguousarray(x[idx], dtype=np.float32)
    t0 = time.perf_counter()
    centroids = train_centroids(
        train_x, n_clusters=n_clusters, niter=IVF_DEFAULT_NITER, nredo=nredo, seed=seed
    )
    fit_s = time.perf_counter() - t0
    labels, mean_cos = assign(centroids, x)
    return labels, fit_s, mean_cos


def agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    """How much ``candidate`` groups items the way ``reference`` does."""
    pairs = pairwise_scores(reference, candidate)
    everyone = np.ones(len(reference), dtype=bool)
    moved = moved_items(reference, candidate, everyone)
    return {
        'ari': float(adjusted_rand_index(reference, candidate)),
        'pair_precision': pairs['pair_precision'],
        'pair_recall': pairs['pair_recall'],
        'same_cluster_share': 1.0 - moved / len(reference),
    }


def label_quality(labels: np.ndarray, partition: np.ndarray) -> dict[str, float | None]:
    scores = pairwise_scores(labels, partition)
    minority = scores['minority_item_rate']
    return {
        'purity': None if minority is None else 1.0 - minority,
        'false_merge_rate': scores['false_merge_rate'],
        'pair_recall': scores['pair_recall'],
    }


def judge(
    incumbent: dict[str, Any],
    candidate: dict[str, Any],
    *,
    ari: float,
    ari_floor: float,
) -> dict[str, Any]:
    """The gate for one candidate on one seed; label checks are skipped (``None``) without labels."""
    checks: dict[str, bool | None] = {
        'ari_not_below_noise_floor': ari >= ari_floor - ARI_BELOW_FLOOR_MAX
    }
    inc_p, cand_p = incumbent.get('purity'), candidate.get('purity')
    inc_f, cand_f = incumbent.get('false_merge_rate'), candidate.get('false_merge_rate')
    checks['purity_within_half_point'] = (
        None if inc_p is None or cand_p is None else cand_p >= inc_p - PURITY_DROP_MAX
    )
    checks['false_merge_within_half_point'] = (
        None if inc_f is None or cand_f is None else cand_f <= inc_f + FALSE_MERGE_RISE_MAX
    )
    return {'checks': checks, 'passes': all(v is not False for v in checks.values())}


def judge_mean(rows: list[dict[str, Any]], *, ari_floor: float) -> dict[str, Any]:
    """The verdict for one candidate: ``judge`` over the seed means of its ``rows``."""

    def mean(key: str, source: str) -> float | None:
        xs = [r[source].get(key) for r in rows if r[source].get(key) is not None]
        return float(np.mean(xs)) if xs else None

    cand = {k: mean(k, 'labels') for k in ('purity', 'false_merge_rate')}
    inc = {k: mean(k, 'incumbent_same_seed_labels') for k in ('purity', 'false_merge_rate')}
    ari = float(np.mean([r['vs_incumbent_seed42']['ari'] for r in rows]))
    return {
        **judge(inc, cand, ari=ari, ari_floor=ari_floor),
        'seed_mean': cand,
        'incumbent_seed_mean': inc,
        'mean_ari': ari,
    }


def read_labels(
    meta_path: Path, labels_path: Path | None, meta_field: str | None
) -> np.ndarray | None:
    """One label per vector row (empty string where the row has none), or ``None``."""
    meta = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if meta_field:
        return np.array([str(m.get(meta_field) or '') for m in meta])
    if labels_path is None:
        return None
    by_id: dict[str, str] = {}
    for line in labels_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_id[row['item_id']] = str(row['label'])
    return np.array([by_id.get(m['crop_id'], '') for m in meta])


def score_labels(labels: np.ndarray | None, partition: np.ndarray) -> dict[str, Any]:
    if labels is None:
        return {}
    known = labels != ''
    return label_quality(labels[known], partition[known])


def run(args: argparse.Namespace) -> dict[str, Any]:
    x = np.load(args.vectors).astype(np.float32)
    x = np.ascontiguousarray(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12))
    labels = None
    if args.meta is not None:
        labels = read_labels(args.meta, args.labels, args.meta_label_field)
    seeds = [PRODUCTION_SEED, *args.seeds]
    out: dict[str, Any] = {
        'vectors': args.vectors.name if args.public else 'private',
        'n': int(x.shape[0]),
        'dim': int(x.shape[1]),
        'labelled': None if labels is None else int((labels != '').sum()),
        'incumbent_nredo': args.incumbent,
        'loadavg_start': os.getloadavg(),
    }
    runs: dict[tuple[int, int], tuple[np.ndarray, float, float]] = {}
    for nredo in [args.incumbent, *args.candidates]:
        for seed in seeds:
            runs[(nredo, seed)] = fit_partition(x, nredo=nredo, seed=seed)
            print(f'fitted nredo={nredo} seed={seed} in {runs[(nredo, seed)][1]:.1f}s', flush=True)
    base = runs[(args.incumbent, PRODUCTION_SEED)][0]
    floors = [float(adjusted_rand_index(base, runs[(args.incumbent, s)][0])) for s in seeds[1:]]
    floor = min(floors) if floors else 1.0
    out['noise_floor_ari_incumbent_seeds'] = floors
    table: dict[str, Any] = {}
    for nredo in [args.incumbent, *args.candidates]:
        per_seed = []
        for seed in seeds:
            part, fit_s, mean_cos = runs[(nredo, seed)]
            row: dict[str, Any] = {
                'seed': seed,
                'fit_s': round(fit_s, 2),
                'mean_cosine_to_centroid': round(mean_cos, 5),
            }
            row['vs_incumbent_seed42'] = agreement(base, part)
            row['labels'] = score_labels(labels, part)
            inc = score_labels(labels, runs[(args.incumbent, seed)][0])
            row['incumbent_same_seed_labels'] = inc
            if nredo != args.incumbent:
                row['gate'] = judge(
                    inc, row['labels'], ari=row['vs_incumbent_seed42']['ari'], ari_floor=floor
                )
            per_seed.append(row)
        verdict = None
        if nredo != args.incumbent:
            verdict = judge_mean(per_seed, ari_floor=floor)
        table[str(nredo)] = {'rows': per_seed, 'gate_on_seed_means': verdict}
    out['by_nredo'] = table
    out['loadavg_end'] = os.getloadavg()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--vectors', type=Path, required=True, help='.npy, rows are the crops')
    ap.add_argument('--meta', type=Path, help='jsonl aligned with the vectors (crop_id, ...)')
    ap.add_argument('--labels', type=Path, help='jsonl item_id/label (joined on crop_id)')
    ap.add_argument('--meta-label-field', help='use this meta field as the label')
    ap.add_argument('--incumbent', type=int, default=5)
    ap.add_argument('--candidates', default='1,2,3')
    ap.add_argument('--seeds', default='7,13', help='extra seeds besides the production 42')
    ap.add_argument('--public', action='store_true', help='record the vectors file name')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.candidates = [int(c) for c in args.candidates.split(',') if c]
    args.seeds = [int(s) for s in args.seeds.split(',') if s]
    result = run(args)
    args.out.write_text(json.dumps(result, indent=2, default=float) + '\n')
    for nredo, entry in result['by_nredo'].items():
        print(nredo, 'gate_on_seed_means=', (entry['gate_on_seed_means'] or {}).get('passes'))
    return 0


if __name__ == '__main__':
    sys.exit(main())

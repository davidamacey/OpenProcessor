"""Run the diagnostics, the configuration baseline (E1) and the class-prior test (E5).

    run.py diag --bundle STEM --labels L.jsonl --coco-gt GT.json --fit-pool POOL.npy --out OUT.json
    run.py e1   --bundle STEM --labels L.jsonl --coco-gt GT.json --out OUT.json
    run.py e5   --bundle STEM --labels L.jsonl --coco-gt GT.json --out OUT.json

``STEM`` is the path stem written by ``fetch_vectors.py``. ``--fit-pool`` is an independent set
of vectors (another project of the same model) used only to fit the centring, whitening and CSLS
statistics, never scored. Every number comes from a seeded, CPU-only computation; the research
environment needs ``igraph`` and ``leidenalg`` for the Leiden and ECG arms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.parity_metrics import adjusted_rand_index
from scripts.research.cluster_diag import diagnostics as dg, heldout as ho, partitions as pt
from scripts.research.cluster_diag.dataset import Bundle, load_bundle


SEED = 20261029
RECALL_TARGETS = (0.15, 0.2, 0.3, 0.5)
LEIDEN_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)


# ---------------------------------------------------------------- diag


def similarity_variants(vectors: Any, pool: Any) -> dict[str, Any]:
    """The similarity matrices of note section 4 (statistics fitted on the independent pool)."""
    mu = pool.mean(0)

    def whiten(dim: int, power: float = 0.5) -> Any:
        m, proj = dg.pca_whiten_fit(pool, dim, power)
        z = dg.pca_whiten_apply(vectors, m, proj)
        return z @ z.T

    centred = dg.l2_normalize(vectors - mu)
    pool_centred = dg.l2_normalize(pool - mu)
    return {
        'raw': vectors @ vectors.T,
        'centered(P mean)': centred @ centred.T,
        'pca_whiten_256': whiten(256),
        'pca_whiten_128': whiten(128),
        'pca_256_nowhiten': whiten(256, 0.0),
        'csls_k10': dg.csls_similarity(vectors, pool, 10),
        'csls_k10_centered': dg.csls_similarity(centred, pool_centred, 10),
        'kreciprocal': dg.kreciprocal_similarity(vectors),
        'dba_k5': dg.dba_similarity(vectors, 5),
    }


def run_diag(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    pool = dg.l2_normalize(np.load(args.fit_pool))
    variants = similarity_variants(b.vectors, pool)
    strata = b.strata_masks()
    out: dict[str, Any] = {
        'geometry_crops_raw': dg.geometry(b.vectors),
        'geometry_fit_pool_raw': dg.geometry(pool[: args.pool_geometry_rows]),
        'knn': {},
        'label_facts': {
            'n_labelled': len(b.truth),
            'strata_counts': {k: int(v.sum()) for k, v in strata.items()},
            'elongated_frac': float(strata['elongated>2:1'].mean()),
            'aspect_gt3_frac': float((np.abs(np.log(b.aspect)) > np.log(3)).mean()),
            'side_lt32_frac': float(strata['side<32px'].mean()),
            'detector_class_agrees_with_label': float(
                np.mean(
                    [
                        b.meta[i]['detector_class_name'] == t
                        for i, t in zip(b.labelled, b.truth, strict=False)
                    ]
                )
            ),
        },
    }
    lab = b.labelled
    for name, sim in variants.items():
        report = dg.knn_report(sim[np.ix_(lab, lab)], b.truth, strata)
        report.update(
            {k: v for k, v in dg.hubness(sim, 10).items() if k.startswith(('hub', 'antihub'))}
        )
        out['knn'][name] = report
    summary: dict[str, Any] = {}
    for name in (
        'raw',
        'centered(P mean)',
        'pca_whiten_256',
        'csls_k10_centered',
        'kreciprocal',
        'dba_k5',
    ):
        sim = variants[name]
        summary[name] = {
            'ahc_average': dg.best_precision_summary(dg.ahc_curve(sim, b.truth, lab, 'average')),
            'ahc_complete': dg.best_precision_summary(dg.ahc_curve(sim, b.truth, lab, 'complete')),
            'leiden_cpm_mknn30': dg.best_precision_summary(dg.leiden_curve(sim, b.truth, lab)),
        }
    out['oracle_curve_summary'] = summary
    return out


# ---------------------------------------------------------------- e1


def _arm_candidates(b: Bundle) -> dict[str, list[tuple[Any, Any]]]:
    """Every arm as ``(parameter, partition over the labelled crops)`` candidates."""
    sim = b.vectors @ b.vectors.T
    lab = b.labelled
    arms: dict[str, list[tuple[Any, Any]]] = {
        'ivf_current(K=min(512,n//6))': [('fixed', pt.ivf_current(b.vectors)[lab])],
        'minibatch_kmeans_2b': [('fixed', pt.region_kmeans(b.vectors)[lab])],
        'ahc_average': [(t, p[lab]) for t, p in pt.ahc_cuts(sim, 'average')],
        'ahc_complete': [(t, p[lab]) for t, p in pt.ahc_cuts(sim, 'complete')],
        'leiden_cpm_mknn30': [
            (f, dg.leiden_cpm_partition(sim, 30, f)[lab]) for f in LEIDEN_FRACTIONS
        ],
        'ecg_mknn30(once)': [('fixed', pt.ecg_partition(sim)[lab])],
    }
    return arms


def _strata_at(part: Any, b: Bundle, held: NDArray[np.bool_]) -> dict[str, dict[str, float | None]]:
    out = {}
    for name, mask in b.strata_masks().items():
        m = mask & held
        if m.sum() > 1:
            p, r, u = ho.score_on(part, b.truth, m)
            out[name] = {'n': int(m.sum()), 'precision': p, 'recall': r, 'unassigned': u}
    return out


def _permutation_ari(fit: Callable[[Any], Any], b: Bundle, *, sort_by_id: bool = False) -> float:
    """ARI between a fit on the exported order and on a seeded permutation of the rows.

    With ``sort_by_id`` both orders are first sorted by crop id, which is what the fetch does
    since #372, so the permutation is undone before the fit sees it.
    """
    n = len(b.vectors)
    order = np.random.default_rng(SEED).permutation(n)
    ids = np.array([m['crop_id'] for m in b.meta])
    base = fit(b.vectors)
    if sort_by_id:
        by_id = np.argsort(ids[order], kind='stable')
        fitted = np.empty(n, dtype=np.int64)
        fitted[order[by_id]] = fit(b.vectors[order[by_id]])
    else:
        fitted = np.empty(n, dtype=np.int64)
        fitted[order] = fit(b.vectors[order])
    return float(adjusted_rand_index(base, fitted))


def run_e1(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    first, _ = ho.split_groups(b.group, SEED)
    half = ho.half_mask(b.group, first)
    out: dict[str, Any] = {
        'seed': SEED,
        'n_crops': len(b.vectors),
        'n_labelled': len(b.truth),
        'labelled_per_half': [int(half.sum()), int((~half).sum())],
        'arms': {},
    }
    for name, cands in _arm_candidates(b).items():
        lookup = dict(cands)
        targets = RECALL_TARGETS if len(cands) > 1 else (0.0,)
        rows = {}
        for target in targets:
            folds = ho.two_fold(cands, b.truth, half, target)
            for fold in folds:
                if fold is not None:
                    held = ~half if fold is folds[0] else half
                    fold['strata'] = _strata_at(lookup[fold['parameter']], b, held)
            rows[f'target_recall>={target}'] = folds
        out['arms'][name] = rows
    sim = b.vectors @ b.vectors.T
    n_cuts = pt.ahc_cuts(sim, 'average')
    cut = n_cuts[int(0.8 * len(n_cuts))][0]
    out['row_permutation_ari'] = {
        'ivf_current_exported_order': _permutation_ari(pt.ivf_current, b),
        'ivf_current_sorted_by_id(#372 fix)': _permutation_ari(pt.ivf_current, b, sort_by_id=True),
        'minibatch_kmeans_2b': _permutation_ari(pt.region_kmeans, b),
        'ahc_average_fixed_cut': _permutation_ari(
            lambda v: fcluster(dg.linkage_of(v @ v.T, 'average'), cut, 'distance'), b
        ),
        'leiden_cpm_fraction_0.3': _permutation_ari(
            lambda v: dg.leiden_cpm_partition(v @ v.T, 30, 0.3), b
        ),
        'ecg_once': _permutation_ari(lambda v: pt.ecg_partition(v @ v.T), b),
    }
    seeds = [dg.leiden_cpm_partition(sim, 30, 0.3, seed=s) for s in range(4)]
    out['leiden_cpm_fraction_0.3_seed_ari'] = [
        float(adjusted_rand_index(seeds[0], other)) for other in seeds[1:]
    ]
    return out


# ---------------------------------------------------------------- e5


def run_e5(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    first, _ = ho.split_groups(b.group, SEED)
    half_all = ho.half_mask(b.group, first)
    conf = np.array([m.get('detector_confidence') or 0.0 for m in b.meta])
    det = np.array([m['detector_class_name'] for m in b.meta])
    out: dict[str, Any] = {'seed': SEED, 'cuts': {}}
    for cut in args.confidence_cuts:
        rows = np.nonzero(conf < cut)[0]
        pos = {int(r): k for k, r in enumerate(rows)}
        lab_in = [k for k, r in enumerate(b.labelled) if int(r) in pos]
        sub_lab = np.array([pos[int(b.labelled[k])] for k in lab_in])
        truth = b.truth[lab_in]
        half = half_all[lab_in]
        v = b.vectors[rows]
        sim = v @ v.T
        z = dg.linkage_of(sim, 'average')
        cuts = np.quantile(z[:, 2], np.linspace(0.5, 0.995, 60))
        per_class = pt.group_linkages(det[rows], sim)
        cands = {
            'global_average_ahc': [
                (float(t), (fcluster(z, t, 'distance') - 1)[sub_lab]) for t in cuts
            ],
            'within_detector_class_average_ahc': [
                (float(t), pt.cut_groups(len(rows), per_class, float(t))[sub_lab]) for t in cuts
            ],
        }
        agree = float(np.mean([det[b.labelled[k]] == b.truth[k] for k in lab_in]))
        entry: dict[str, Any] = {
            'subset_crops': len(rows),
            'subset_labelled': len(lab_in),
            'labelled_per_half': [int(half.sum()), int((~half).sum())],
            'detector_class_agrees_with_label': agree,
            'methods': {},
        }
        for name, c in cands.items():
            entry['methods'][name] = {
                f'target_recall>={t}': ho.two_fold(c, truth, half, t) for t in RECALL_TARGETS
            }
        out['cuts'][f'detector_confidence<{cut}'] = entry
    return out


# ---------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('diag', 'e1', 'e5'):
        p = sub.add_parser(name)
        p.add_argument('--bundle', type=Path, required=True)
        p.add_argument('--labels', type=Path, required=True)
        p.add_argument('--coco-gt', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        if name == 'diag':
            p.add_argument('--fit-pool', type=Path, required=True)
            p.add_argument('--pool-geometry-rows', type=int, default=20_000)
        if name == 'e5':
            p.add_argument('--confidence-cuts', type=float, nargs='+', default=[0.5, 0.7])
    args = ap.parse_args(argv)
    result = {'diag': run_diag, 'e1': run_e1, 'e5': run_e5}[args.cmd](args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    print(json.dumps(result, indent=1, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())

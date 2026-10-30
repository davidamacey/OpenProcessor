"""FG-2 and FG-4 inside the car key: grouping methods on one representation, both folds.

    python -m scripts.research.fine_grained.group_run --emb DIR --out FILE.json \
        --model pe_core_l14_336 --variant centre --methods ahc kr dba leiden ecg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster

from scripts.bench.cluster_eval_metrics import assignment_stability
from scripts.research.cluster_diag.diagnostics import linkage_of
from scripts.research.fine_grained import bars, group_eval as ge, methods
from scripts.research.fine_grained.data import load_universe, load_vectors, restrict, run_info


def _ahc_at(sim: np.ndarray, height: float) -> np.ndarray:
    return np.asarray(fcluster(linkage_of(sim, 'average'), height, 'distance') - 1)


SIMS = {
    'ahc': methods.cos_sim,
    'kr': lambda v: methods.kreciprocal_similarity(v.astype(np.float32)),
    'dba': lambda v: methods.dba_similarity(v.astype(np.float32)),
}


def family(method: str, vec: np.ndarray) -> tuple[list[float], list[np.ndarray], np.ndarray]:
    sim = SIMS[method](vec) if method in SIMS else methods.cos_sim(vec)
    if method in SIMS:
        return (*ge.ahc_family(sim), sim)
    if method == 'leiden':
        return (*ge.leiden_family(sim), sim)
    return (*ge.ecg_family(sim), sim)


def rebuild(method: str, vec: np.ndarray, param: float, seed: int = 0) -> np.ndarray:
    sim = SIMS[method](vec) if method in SIMS else methods.cos_sim(vec)
    if method in SIMS:
        return _ahc_at(sim, param)
    if method == 'leiden':
        return methods.leiden_cpm_partition(sim, 30, param, seed=seed)
    return methods.ecg_partition(sim, 30, seed=seed)


def stability(method: str, vec: np.ndarray, param: float, part: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        'permutation_ari': ge.permutation_ari(lambda v: rebuild(method, v, param), vec, part)
    }
    if method in ('leiden', 'ecg'):
        others = [rebuild(method, vec, param, seed=s) for s in ge.STABILITY_SEEDS[1:]]
        aris = [assignment_stability(part, o)['ari'] for o in others]
        out['seed_ari_min'] = min(a for a in aris if a is not None) if aris else None
    return out


def run_method(
    method: str, vec: np.ndarray, u: Any, ahc_curves: dict[int, list[dict[str, Any]]] | None
) -> dict[str, Any]:
    t0 = time.time()
    params, parts, _ = family(method, vec)
    res: dict[str, Any] = {'seconds_family': time.time() - t0, 'folds': {}}
    for f in (1, 2):
        read = ge.read_fold(parts, params, u, f)
        if method == 'ahc':
            ahc_curves = ahc_curves if ahc_curves is not None else {}
            ahc_curves[f] = read['test_curve']
        elif ahc_curves and f in ahc_curves:
            read['vs_ahc'] = ge.matched_vs_ahc(read, ahc_curves[f])
        res['folds'][f'fold{f}'] = {k: v for k, v in read.items() if k != 'part_index'}
        if 'part_index' in read and f == 1:
            idx = read['part_index']
            res['stability'] = stability(method, vec, params[idx], parts[idx])
    if method != 'ahc':
        folds = [res['folds'][f'fold{f}'].get('vs_ahc', {}) for f in (1, 2)]
        needs = method in ('leiden', 'ecg')
        res['fg4'] = bars.fg4(folds, res.get('stability', {}).get('seed_ari_min'), needs)
    res['_ahc_curves'] = ahc_curves
    return res


def run(args: argparse.Namespace) -> dict[str, Any]:
    emb = Path(args.emb)
    u = load_universe(emb)
    vec = load_vectors(emb, args.model, args.variant)
    if args.distractors is not None:
        u, vec = restrict(u, vec, args.distractors)
    out: dict[str, Any] = {
        **run_info(),
        'model': args.model,
        'variant': args.variant,
        'rows': len(vec),
        'distractors': args.distractors,
        'methods': {},
    }
    curves: dict[int, list[dict[str, Any]]] | None = None
    for method in args.methods:
        res = run_method(method, vec, u, curves)
        curves = res.pop('_ahc_curves')
        out['methods'][method] = res
        print(method, 'done', round(res['seconds_family'], 1), 's', flush=True)
        if method == 'ahc':
            f1 = res['folds']['fold1']
            if 'test' in f1:
                t = f1['test']
                res['fg2'] = bars.fg2(
                    t['model']['bcubed_precision'],
                    t['make']['bcubed_precision'],
                    t['r1'],
                    res.get('stability', {}).get('permutation_ari'),
                )
        Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--variant', default='centre')
    ap.add_argument(
        '--distractors', type=int, default=None, help='cap on distractor rows (graph methods)'
    )
    ap.add_argument('--methods', nargs='+', default=['ahc'])
    args = ap.parse_args()
    result = run(args)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True, default=float))


if __name__ == '__main__':
    main()

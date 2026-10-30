"""FG-1 representation and preprocessing screen: neighbour scores per model and variant, both folds.

    python -m scripts.research.fine_grained.screen_run --emb DIR --out FILE.json \
        --models pe_core_l14_336 ... --variants centre
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.fine_grained import bars
from scripts.research.fine_grained.data import Universe, load_universe, load_vectors, run_info
from scripts.research.fine_grained.metrics import ranking_scores


LEVELS = ('make', 'model', 'generation')


def _labels(u: Universe, level: str, rows: np.ndarray) -> np.ndarray:
    if level == 'generation':
        return np.array(
            [f'{m}|{g}' for m, g in zip(u.model[rows], u.generation[rows], strict=True)]
        )
    return np.asarray(getattr(u, level)[rows])


def fold_scores(u: Universe, vec: np.ndarray, fold: int) -> dict[str, Any]:
    _, test = u.fold(fold)
    sim = vec[test] @ vec[test].T
    out: dict[str, Any] = {lv: ranking_scores(sim, _labels(u, lv, test)) for lv in LEVELS}
    # the same queries against the labelled test half plus every distractor as a wrong answer
    dis = np.nonzero(~u.labelled)[0]
    rows = np.concatenate([test, dis])
    full = vec[test] @ vec[rows].T
    full[np.arange(len(test)), np.arange(len(test))] = -np.inf
    top = np.argsort(-full, axis=1, kind='stable')[:, :10]
    big = {}
    for lv in ('make', 'model'):
        y = np.concatenate([_labels(u, lv, test), np.array([f'__d{i}' for i in range(len(dis))])])
        big[lv] = {
            'P@10': float((y[top] == y[: len(test), None]).mean()),
            'queries': len(test),
            'gallery': len(rows),
        }
    out['with_distractors'] = big
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    emb = Path(args.emb)
    u = load_universe(emb, args.subset)
    result: dict[str, Any] = {
        **run_info(),
        'models': {},
        'folds': 'fold 1 tests on held-out; fold 2 on calibration',
    }
    for model in args.models:
        for variant in args.variants:
            vec = load_vectors(emb, model, variant, args.subset)
            result['models'][f'{model}__{variant}'] = {
                f'fold{f}': fold_scores(u, vec, f) for f in (1, 2)
            }
    return result


def fg1(result: dict[str, Any], base: str, models: list[str]) -> dict[str, Any]:
    """FG-1 for each challenger against ``base`` (a key of ``result['models']``)."""
    out: dict[str, Any] = {}
    for key in models:
        if key == base:
            continue
        row = {}
        for f in (1, 2):
            c = result['models'][key][f'fold{f}']['model']['P@10']
            b = result['models'][base][f'fold{f}']['model']['P@10']
            row[f'fold{f}'] = {
                'p10_model': c,
                'base': b,
                'delta': c - b,
                'verdict': bars.fg1_challenger(c, b),
            }
        out[key] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--emb', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--variants', nargs='+', default=['centre'])
    ap.add_argument('--subset', choices=['all', 'setc'], default='all')
    ap.add_argument('--base', default='pe_core_l14_336__centre')
    args = ap.parse_args()
    result = run(args)
    if args.base in result['models']:
        result['fg1_challengers'] = fg1(result, args.base, list(result['models']))
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    for k, v in result['models'].items():
        f1 = v['fold1']
        print(
            k,
            {lv: round(f1[lv]['P@10'], 3) for lv in LEVELS},
            'dist',
            {lv: round(x['P@10'], 3) for lv, x in f1['with_distractors'].items()},
        )


if __name__ == '__main__':
    main()

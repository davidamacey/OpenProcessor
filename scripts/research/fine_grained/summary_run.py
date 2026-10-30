"""Assemble the FG-0 to FG-7 verdict table from the result files of the other runs.

    python -m scripts.research.fine_grained.summary_run --res DIR --out FILE.json
Reads (in DIR): fg0_report.json, screen_centre.json, screen_variants.json, screen_whole.json,
class_p10.json, group_*.json, fg6_*.json, fg7_make.json, fg7_model.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.research.fine_grained import bars


def _load(res: Path, name: str) -> dict[str, Any] | None:
    p = res / name
    return json.loads(p.read_text()) if p.exists() else None


def fg1_variants(screen: dict[str, Any], class_p10: dict[str, Any] | None) -> dict[str, Any]:
    """Each non-centre variant against the same model at centre, fold 1 read, fold 2 reported."""
    out: dict[str, Any] = {}
    for key in sorted(screen['models']):
        model, variant = key.split('__')
        if variant == 'centre' or f'{model}__centre' not in screen['models']:
            continue
        row: dict[str, Any] = {}
        for f in (1, 2):
            v = screen['models'][key][f'fold{f}']
            b = screen['models'][f'{model}__centre'][f'fold{f}']
            cls = (class_p10 or {}).get('class_p10', {})
            row[f'fold{f}'] = {
                'model_p10': v['model']['P@10'],
                'base_model_p10': b['model']['P@10'],
                'make_p10': v['make']['P@10'],
                'base_make_p10': b['make']['P@10'],
                'verdict': bars.fg1_variant(
                    v['model']['P@10'],
                    b['model']['P@10'],
                    v['make']['P@10'],
                    b['make']['P@10'],
                    cls.get(key),
                    cls.get(f'{model}__centre'),
                ),
            }
        out[key] = row
    return out


def _ahc_test(group: dict[str, Any], fold: int) -> dict[str, Any] | None:
    test: dict[str, Any] | None = group['methods']['ahc']['folds'][f'fold{fold}'].get('test')
    return test


def fg2_and_fg5(res: Path, challengers: dict[str, Any]) -> dict[str, Any]:
    """FG-2 per representation (fold 1 read, fold 2 reopen check) and FG-5 against PE-Core's R1."""
    groups = {}
    for p in sorted(res.glob('group_*.json')):
        g = json.loads(p.read_text())
        if 'ahc' in g['methods']:
            groups[f'{g["model"]}__{g["variant"]}'] = g
    out: dict[str, Any] = {}
    pe = groups.get('pe_core_l14_336__centre')
    pe_test = None if pe is None else _ahc_test(pe, 1)
    r1_pe = None if pe_test is None else pe_test['r1']
    for key, g in groups.items():
        t1, t2 = _ahc_test(g, 1), _ahc_test(g, 2)
        if t1 is None:
            out[key] = {'FG-2': 'no operating point on calibration'}
            continue
        fg2 = g['methods']['ahc'].get('fg2') or bars.fg2(
            t1['model']['bcubed_precision'],
            t1['make']['bcubed_precision'],
            t1['r1'],
            g['methods']['ahc'].get('stability', {}).get('permutation_ari'),
        )
        row: dict[str, Any] = {'FG-2': fg2}
        if t2 is not None:
            row['reopen_fold2'] = bars.reopen(
                {
                    'model_bcubed_precision': t2['model']['bcubed_precision'],
                    'make_bcubed_precision': t2['make']['bcubed_precision'],
                },
                {
                    'model_bcubed_precision': bars.FG2_MODEL_PRECISION,
                    'make_bcubed_precision': bars.FG2_MAKE_PRECISION,
                },
            )
        ch = challengers.get(key)
        if key != 'pe_core_l14_336__centre' and ch is not None:
            checks = fg2['checks']
            row['FG-5'] = bars.fg5(
                ch['fold1']['verdict'] == bars.PASS,
                t1['r1'],
                r1_pe,
                checks['model_bcubed_precision>=0.90'] and checks['make_bcubed_precision>=0.95'],
            )
        out[key] = row
    return out


def fg7_rows(res: Path) -> dict[str, Any]:
    out = {}
    for level in ('make', 'model'):
        rep = _load(res, f'fg7_{level}.json')
        if rep is None:
            out[level] = {'verdict': bars.INCONCLUSIVE, 'note': 'not run'}
            continue
        j = rep['judges']['haiku-subagent']
        k, p = j['pair']['kappa'], j['cluster']['cell_flags']['precision']
        out[level] = {'pair_kappa': k, 'flag_precision': p, 'verdict': bars.fg7(k, p)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--res', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    res = Path(args.res)
    out: dict[str, Any] = {}
    fg0 = _load(res, 'fg0_report.json')
    if fg0:
        out['FG-0'] = fg0['score']['fg0_verdict']
    merged: dict[str, Any] = {}
    for p in sorted(res.glob('screen_variants*.json')) + sorted(
        res.glob('screen_peg_variants.json')
    ):
        merged.update(json.loads(p.read_text())['models'])
    variants = {'models': merged}
    if merged:
        out['FG-1_variants'] = fg1_variants(variants, _load(res, 'class_p10.json'))
    screen = _load(res, 'screen_centre.json') or {}
    out['FG-2_FG-5'] = fg2_and_fg5(res, screen.get('fg1_challengers', {}))
    out['FG-7'] = fg7_rows(res)
    out['FG-3'] = bars.fg3(None, None, readable=False)
    for p in sorted(res.glob('fg6_*.json')):
        d = json.loads(p.read_text())
        f1 = d['fold1']
        out[f'FG-6_{p.stem[4:]}'] = bars.fg6(
            {m: a for m, a in f1['auroc_by_make'].items() if a is not None},
            f1['pooled_flag_precision'],
        )
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()

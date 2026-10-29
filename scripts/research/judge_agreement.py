#!/usr/bin/env python3
"""Judge agreement report: each judge against ground truth, judges against each other, layouts.

    judge_agreement.py report --sheets DIR --runs run1.jsonl run2.jsonl [--out report.json]

``report`` scores cluster sheets (same/different, flagged cells, naming, judge-estimated purity
with a sheet-level bootstrap interval) and pair sheets (yes/no with "unsure" kept out of the
accuracy but reported), compares judges (pairwise Cohen's, Fleiss'), runs the layout calibration
when the sample holds calibration sheets and the planted-control check when it holds controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.research import judge_sheetdir as sd
from scripts.research.judge_protocol import odd_items
from scripts.research.judge_stats import (
    binary_scores,
    cluster_bootstrap,
    cohens_kappa,
    fleiss_kappa,
    wilson_interval,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

KAPPA_TOLERANCE = 0.05
CONTROL_MAX_ERROR = 0.2
Runs = dict[str, dict[str, dict[str, Any]]]


def load_runs(paths: Iterable[Path]) -> Runs:
    """``{model: {sheet_id: record}}``; the last valid record of a sheet wins."""
    out: Runs = defaultdict(dict)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            doc = json.loads(line)
            if 'sheet_id' not in doc:
                continue
            kept = out[doc['model']].get(doc['sheet_id'])
            if doc.get('error') and kept is not None and not kept.get('error'):
                continue
            out[doc['model']][doc['sheet_id']] = doc
    return dict(out)


def _valid(recs: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {r['sheet_id']: r for r in recs if r.get('verdict') is not None}


def _name_matches(name: str, majority: str) -> bool:
    a, b = name.strip().lower(), majority.strip().lower().replace('_', ' ')
    return bool(a) and (a in b or b in a)


def _cell_counts(truth: Mapping[str, Any], verdict: Mapping[str, Any]) -> tuple[int, int, int]:
    true_odd, got = set(truth['odd_cells']), set(verdict['odd_cells'])
    return len(true_odd & got), len(got - true_odd), len(true_odd - got)


def _purity(n_cells: int, n_odd: int) -> float:
    return 1 - n_odd / n_cells


def cluster_metrics(
    truth_rows: Sequence[Mapping[str, Any]],
    recs: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scored = _valid(recs)
    truth = {t['sheet_id']: t for t in truth_rows if t['layout'] == 'cluster' and t['truth']}
    ids = sorted(set(scored) & set(truth))
    t_diff = [not truth[i]['truth']['same'] for i in ids]
    p_diff = [not scored[i]['verdict']['same'] for i in ids]
    tp = fp = fn = 0
    names = 0
    by_sheet_j: dict[str, list[float]] = {}
    for i in ids:
        a, b, c = _cell_counts(truth[i]['truth'], scored[i]['verdict'])
        tp, fp, fn = tp + a, fp + b, fn + c
        names += _name_matches(scored[i]['verdict'].get('name', ''), truth[i]['truth']['majority'])
        n = len(manifests[i]['cells'])
        by_sheet_j[i] = [_purity(n, len(scored[i]['verdict']['odd_cells']))]
    diff = binary_scores(t_diff, p_diff)
    correct = diff['tp'] + diff['tn']
    t_purity = [
        _purity(len(manifests[i]['cells']), len(truth[i]['truth']['odd_cells'])) for i in ids
    ]
    j_mean = sum(v[0] for v in by_sheet_j.values()) / len(ids) if ids else None
    return {
        'n': len(ids),
        'invalid': sum(1 for r in recs if r.get('verdict') is None),
        'missing': len(
            set(truth) - set(scored) - {r['sheet_id'] for r in recs if r.get('verdict') is None}
        ),
        'different': diff,
        'accuracy_wilson95': wilson_interval(correct, len(ids)),
        'false_same_rate': diff['false_negative_rate'],
        'cell_flags': {
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
        },
        'name_match_rate': names / len(ids) if ids else None,
        'purity': {
            'truth': sum(t_purity) / len(ids) if ids else None,
            'judge': j_mean,
            'judge_ci95': cluster_bootstrap(by_sheet_j, seed=0),
        },
        'by_contamination': _by_bad(ids, truth, scored),
        'cost': _cost(recs),
    }


def _by_bad(
    ids: Sequence[str],
    truth: Mapping[str, Mapping[str, Any]],
    scored: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for i in ids:
        if truth[i]['truth']['same']:
            continue
        key = str(truth[i].get('bad_kind', 'all'))
        groups[key].append(not scored[i]['verdict']['same'])
    return {k: {'n': len(v), 'recall': sum(v) / len(v)} for k, v in sorted(groups.items())}


def _cost(recs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    live = [r for r in recs if not r.get('cached')]
    lat = [r['latency_ms'] for r in live if r.get('latency_ms') is not None]
    return {
        'calls': len(live),
        'input_tokens': sum(r.get('input_tokens') or 0 for r in live),
        'output_tokens': sum(r.get('output_tokens') or 0 for r in live),
        'mean_latency_ms': sum(lat) / len(lat) if lat else None,
    }


def pair_metrics(
    truth_rows: Sequence[Mapping[str, Any]], recs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    scored = _valid(recs)
    truth_t: list[bool] = []
    pred: list[bool] = []
    total = 0
    for t in truth_rows:
        if t['layout'] != 'pair' or t['sheet_id'] not in scored:
            continue
        answers = {r['row']: r['answer'] for r in scored[t['sheet_id']]['verdict']['rows']}
        for row, same in t['truth']['rows'].items():
            total += 1
            if answers[int(row)] != 'unsure':
                truth_t.append(bool(same))
                pred.append(answers[int(row)] == 'yes')
    s = binary_scores([not x for x in truth_t], [not x for x in pred]) if truth_t else {}
    return {
        'rows_total': total,
        'rows_decided': len(truth_t),
        'unsure_rate': (total - len(truth_t)) / total if total else None,
        'accuracy': s.get('accuracy'),
        'accuracy_wilson95': wilson_interval(s['tp'] + s['tn'], s['n']) if s else None,
        'kappa': cohens_kappa(truth_t, pred) if truth_t else None,
        'false_same_rate': s.get('false_negative_rate') if s else None,
        'cost': _cost(recs),
    }


def judge_agreement(
    truth_rows: Sequence[Mapping[str, Any]], by_judge: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    truth = {t['sheet_id']: t for t in truth_rows if t['layout'] == 'cluster' and t['truth']}
    valid = {j: _valid(r.values()) for j, r in by_judge.items()}
    judges = sorted(valid)
    verdict = {
        j: {i: v['verdict']['same'] for i, v in valid[j].items() if i in truth} for j in judges
    }
    out: dict[str, Any] = {'pairwise_kappa': {}, 'kappa_with_truth': {}, 'fleiss_kappa': None}
    for x, a in enumerate(judges):
        common = sorted(verdict[a])
        out['kappa_with_truth'][a] = cohens_kappa(
            [truth[i]['truth']['same'] for i in common], [verdict[a][i] for i in common]
        )
        for b in judges[x + 1 :]:
            both = sorted(set(verdict[a]) & set(verdict[b]))
            out['pairwise_kappa'][f'{a}|{b}'] = cohens_kappa(
                [verdict[a][i] for i in both], [verdict[b][i] for i in both]
            )
    if len(judges) >= 2:
        common = sorted(set.intersection(*(set(v) for v in verdict.values())))
        out['fleiss_kappa'] = fleiss_kappa([[verdict[j][i] for j in judges] for i in common])
        out['common_sheets'] = len(common)
    return out


def recommend_layout(kappas: Mapping[tuple[int, int], float]) -> tuple[int, int]:
    """Smallest (cells, thumb_px) whose kappa is within 0.05 of the best layout."""
    best = max(kappas.values())
    return min((k for k, v in kappas.items() if v >= best - KAPPA_TOLERANCE))


def _layout_key(t: Mapping[str, Any]) -> str:
    return f'{t["grid"]}@{t["thumb_px"]}'


def _shown(manifest: Mapping[str, Any]) -> set[str]:
    return {c['item_id'] for c in manifest['cells']}


def _flagged(manifest: Mapping[str, Any], verdict: Mapping[str, Any]) -> set[str]:
    return {item for item, _ in odd_items(manifest, verdict)}


def calibration(
    truth_rows: Sequence[Mapping[str, Any]], runs: Runs, manifests: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Per judge: kappa with truth per layout, and stability against the largest layout."""
    cal = [t for t in truth_rows if 'grid' in t]
    cells_of = {
        _layout_key(t): int(t['grid'].split('x')[0]) * int(t['grid'].split('x')[1]) for t in cal
    }
    ref = max(cells_of, key=lambda k: (cells_of[k], int(k.split('@')[1])))
    out: dict[str, Any] = {}
    for judge, recs in runs.items():
        valid = _valid(recs.values())
        layouts: dict[str, Any] = {}
        stability: dict[str, Any] = {}
        by_case: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for t in cal:
            if t['sheet_id'] in valid:
                by_case[t['case']][_layout_key(t)] = t
        for key in cells_of:
            rows = [c[key] for c in by_case.values() if key in c]
            truth_same = [r['truth']['same'] for r in rows]
            got = [valid[r['sheet_id']]['verdict']['same'] for r in rows]
            layouts[key] = {
                'n': len(rows),
                'kappa_truth': cohens_kappa(truth_same, got),
                'accuracy': sum(a == b for a, b in zip(truth_same, got, strict=True)) / len(rows)
                if rows
                else None,
            }
            stability[key] = _stability(by_case, key, ref, valid, manifests)
        usable = {
            (cells_of[k], int(k.split('@')[1])): v['kappa_truth']
            for k, v in layouts.items()
            if v['kappa_truth'] is not None
        }
        out[judge] = {
            'layouts': layouts,
            'stability': stability,
            'reference': ref,
            'recommended': recommend_layout(usable) if usable else None,
        }
    return out


def _stability(
    by_case: Mapping[str, Mapping[str, Mapping[str, Any]]],
    key: str,
    ref: str,
    valid: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    a: list[bool] = []
    b: list[bool] = []
    overlaps: list[float] = []
    for layouts in by_case.values():
        if key not in layouts or ref not in layouts:
            continue
        ra, rb = layouts[key]['sheet_id'], layouts[ref]['sheet_id']
        a.append(valid[ra]['verdict']['same'])
        b.append(valid[rb]['verdict']['same'])
        both = _shown(manifests[ra]) & _shown(manifests[rb])
        fa = _flagged(manifests[ra], valid[ra]['verdict']) & both
        fb = _flagged(manifests[rb], valid[rb]['verdict']) & both
        overlaps.append(len(fa & fb) / len(fa | fb) if fa | fb else 1.0)
    return {
        'n': len(a),
        'same_kappa_vs_reference': cohens_kappa(a, b),
        'flag_overlap': sum(overlaps) / len(overlaps) if overlaps else None,
    }


def controls_report(
    truth_rows: Sequence[Mapping[str, Any]], recs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """False-accept / false-reject on planted controls; a failing batch is to be discarded."""
    valid = _valid(recs.values())
    ctl = [t for t in truth_rows if t.get('control') and t['sheet_id'] in valid]
    diff = [t for t in ctl if t['control'] == 'different']
    same = [t for t in ctl if t['control'] == 'same']
    fa = sum(valid[t['sheet_id']]['verdict']['same'] for t in diff)
    fr = sum(not valid[t['sheet_id']]['verdict']['same'] for t in same)
    far = fa / len(diff) if diff else None
    frr = fr / len(same) if same else None
    return {
        'n_controls': len(ctl),
        'false_accept_rate': far,
        'false_reject_rate': frr,
        'batch_ok': all(x is None or x <= CONTROL_MAX_ERROR for x in (far, frr)) and bool(ctl),
    }


def order_effect(recs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Agreement between a sheet and its re-ordered ``-r`` twin (same members, other order)."""
    valid = _valid(recs.values())
    pairs = [(i, f'{i}-r') for i in valid if f'{i}-r' in valid]
    same_verdict = [valid[a]['verdict']['same'] == valid[b]['verdict']['same'] for a, b in pairs]
    return {
        'n_pairs': len(pairs),
        'verdict_flip_rate': 1 - sum(same_verdict) / len(pairs) if pairs else None,
    }


def disagreements(
    truth_rows: Sequence[Mapping[str, Any]],
    recs: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Cluster sheets where the judge's same/different or flagged items differ from the truth."""
    scored = _valid(recs)
    out: list[dict[str, Any]] = []
    for t in truth_rows:
        r = scored.get(t['sheet_id'])
        if r is None or t['layout'] != 'cluster' or not t['truth']:
            continue
        manifest = manifests[t['sheet_id']]
        true_items = {
            c['item_id'] for c in manifest['cells'] if c['cell'] in t['truth']['odd_cells']
        }
        got = _flagged(manifest, r['verdict'])
        if got != true_items:
            out.append(
                {
                    'sheet_id': t['sheet_id'],
                    'truth_same': t['truth']['same'],
                    'judge_same': r['verdict']['same'],
                    'missed_items': sorted(true_items - got),
                    'false_flags': sorted(got - true_items),
                }
            )
    return out


def build_report(sheets: Path, runs: Runs) -> dict[str, Any]:
    truth_rows = list(sd.load_truth(sheets).values())
    manifests = sd.load_manifests(sheets)
    report: dict[str, Any] = {'sheets': str(sheets), 'spec': sd.load_spec(sheets), 'judges': {}}
    for judge, recs in sorted(runs.items()):
        rec_list = list(recs.values())
        entry: dict[str, Any] = {
            'cluster': cluster_metrics(
                [t for t in truth_rows if 'grid' not in t], rec_list, manifests
            ),
            'disagreements': disagreements(
                [t for t in truth_rows if 'grid' not in t], rec_list, manifests
            ),
            'pair': pair_metrics(truth_rows, rec_list),
            'order_effect': order_effect(recs),
        }
        if any(t.get('control') for t in truth_rows):
            entry['controls'] = controls_report(truth_rows, recs)
        report['judges'][judge] = entry
    report['agreement'] = judge_agreement([t for t in truth_rows if 'grid' not in t], runs)
    if any('grid' in t for t in truth_rows):
        report['calibration'] = calibration(truth_rows, runs, manifests)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    rp = sub.add_parser('report')
    rp.add_argument('--sheets', required=True)
    rp.add_argument('--runs', nargs='+', required=True)
    rp.add_argument('--out')
    args = ap.parse_args(argv)
    report = build_report(Path(args.sheets), load_runs(Path(p) for p in args.runs))
    text = json.dumps(report, indent=1, sort_keys=True, default=list)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

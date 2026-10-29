"""Experiments Z1 and Z2 (decision note section 0.4): the loop-gate simulator on three partitions.

    z_run.py z1 --set NAME BUNDLE LABELS COCO_GT SPLIT [--set ...] --out OUT.json
    z_run.py z2 --bundle STEM --labels L --coco-gt G --split-manifest S --out OUT.json
    z_run.py z3 --bundle STEM --labels L --coco-gt G --split-manifest S --z2 Z2.json --out OUT.json

Arms: ``ivf`` (production IVF refit on the same vectors), ``key_only`` (one cluster per detector
key) and ``within_key`` (average linkage inside each key at the cut the section-6 operating-point
rule chooses on the calibration half). Z1 is a code and direction check on sets already read; Z2
reads the U bars once, fold 1, on the held-out half of a fresh set and shows A1 to A7 beside them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import scorecard
from scripts.datasets.vehicle_eval_selection import OPENSET
from scripts.research.cluster_diag import (
    acceptance as acc,
    class_prior as cp,
    gate_sim as gs,
    merge_stage as ms,
    partitions as pt,
    u_bars as ub,
    y_acceptance as ya,
)
from scripts.research.cluster_diag.dataset import Bundle, detector_keys, load_bundle
from scripts.research.cluster_diag.run import SEED, calibration_mask


SIM_SEEDS = tuple(range(20))  # sampling seeds of the gate; every figure is the mean over them


def arm_partitions(
    b: Bundle, keys: Any, param: float, arm: ya.Arm, incumbent: Any
) -> dict[str, Any]:
    """Cluster codes of the labelled crops for the three arms (IVF refit, key alone, within-key)."""
    return {
        'ivf': np.asarray(incumbent),
        'key_only': cp.key_only_partition(np.asarray(keys)[b.labelled]),
        'within_key': np.asarray(arm.full(param)[b.labelled]),
    }


def gate_on_pool(
    parts: dict[str, Any], truth: Any, pool: Any, *, noisy: bool = True
) -> dict[str, Any]:
    """The simulator on the ``pool`` mask of labelled crops, every arm; a clean row, a noisy row
    and the cluster-size reach."""
    t = np.asarray(truth)[pool]
    out: dict[str, Any] = {'items': int(pool.sum())}
    for name, codes in parts.items():
        c = np.asarray(codes)[pool]
        sizes = np.unique(c, return_counts=True)[1].tolist()
        row = {
            'reach': gs.gate_reach(sizes),
            'clean': gs.seeded_runs(c, t, seeds=SIM_SEEDS, open_set=OPENSET),
        }
        if noisy:
            row['noisy_0.10'] = gs.seeded_runs(
                c, t, seeds=SIM_SEEDS, open_set=OPENSET, noise=gs.NOISE
            )
        out[name] = row
    return out


def _incumbent(b: Bundle) -> Any:
    return pt.ivf_current(b.vectors)[b.labelled]


def run_z1(args: argparse.Namespace) -> dict[str, Any]:
    """Direction and code check on sets already read. Never acceptance."""
    out: dict[str, Any] = {
        'label': 'code check and direction only; both sets were read by earlier waves',
        'gate': {'theta': gs.THETA, 'alpha': gs.ALPHA, 'max_dissent': gs.MAX_DISSENT},
        'sim_seeds': list(SIM_SEEDS),
        'sets': {},
    }
    for name, bundle, labels, coco_gt, split in args.set:
        b = load_bundle(Path(bundle), Path(labels), Path(coco_gt))
        keys = detector_keys(b.meta)
        fit = calibration_mask(b, Path(split))
        arm = ya.global_cut_arm(b, keys)
        op = acc.operating_point(arm.candidates, b.truth, fit)
        if op is None:
            out['sets'][name] = {'operating_point': None}
            continue
        parts = arm_partitions(b, keys, float(op['parameter']), arm, _incumbent(b))
        out['sets'][name] = {
            'operating_point': op,
            'crops': len(b.vectors),
            'labelled': len(b.truth),
            'heldout_pool': gate_on_pool(parts, b.truth, ~fit),
            'all_labelled_pool': gate_on_pool(parts, b.truth, np.ones(len(b.truth), bool)),
        }
    return out


def run_z2(args: argparse.Namespace) -> dict[str, Any]:
    """The U bars once on the held-out half (fold 1) with A1 to A7 beside them."""
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys = detector_keys(b.meta)
    first = calibration_mask(b, args.split_manifest)
    incumbent = _incumbent(b)
    arm = ya.global_cut_arm(b, keys)
    ids = [m['crop_id'] for m in b.meta]
    folds = {
        fold: ya.evaluate_fold(b, keys, arm, incumbent, fit, ids=ids, seed=SEED)
        for fold, fit in (
            ('fold1_calibration_to_heldout', first),
            ('fold2_heldout_to_calibration', ~first),
        )
    }
    f1 = folds['fold1_calibration_to_heldout']
    out: dict[str, Any] = {
        'registered': {
            'u_a': ub.U2_BAR,
            'u_b_factor': ub.IVF_FACTOR,
            'u_c_factor': ub.KEY_FACTOR,
            'u_d': ub.OPEN_BAR,
            'min_open_propagated': ub.MIN_OPEN_PROPAGATED,
            'sim_seeds': list(SIM_SEEDS),
            'noise': gs.NOISE,
            'gate': {'theta': gs.THETA, 'alpha': gs.ALPHA, 'max_dissent': gs.MAX_DISSENT},
            'open_set_classes': sorted(OPENSET),
        },
        'crops': len(b.vectors),
        'labelled': len(b.truth),
        'detector_keys': len(set(keys.tolist())),
        'heldout_labelled': int((~first).sum()),
        'calibration_labelled': int(first.sum()),
        'acceptance_A1_A7': folds,
    }
    if f1['operating_point'] is None:
        out['u_bars'] = {'verdict': ub.FAIL, 'reason': 'no operating point on the calibration half'}
        return out
    parts = arm_partitions(b, keys, float(f1['parameter']), arm, incumbent)
    pool = gate_on_pool(parts, b.truth, ~first)
    clean = {k: pool[k]['clean'] for k in parts}
    rows = ub.u_bars(clean['within_key'], clean['ivf'], clean['key_only'])
    out['cut'] = f1['parameter']
    out['heldout_pool'] = pool
    out['u_bars'] = {
        'rows': rows,
        'verdict': ub.overall(rows),
        'cheaper_simple_design_at_the_floor': ub.simpler_winner(clean),
    }
    out['supplementary_all_labelled_pool'] = gate_on_pool(
        parts, b.truth, np.ones(len(b.truth), bool), noisy=False
    )
    return out


WINDOWS = (0.05, 0.10, 0.15)  # candidate heights above the cut that the merge stage may reach
ADOPT_FACTOR = 0.9  # Z3: total decisions, pair checks included, at most this share of no-merge's
CHECK_COST = 1.0  # decisions one pair check costs


def _merge_arm(
    linkages: Any,
    b: Bundle,
    pool: Any,
    cut: float,
    window: float,
    judge: ms.Judge,
    check_cost: float,
) -> dict[str, Any]:
    """The gate on the merged within-key partition of the ``pool`` mask, mean over sampling seeds
    (each seed also draws its own judge answers)."""
    row_label: list[str | None] = [None] * len(b.vectors)
    for i, row in enumerate(b.labelled):
        if pool[i]:
            row_label[int(row)] = str(b.truth[i])
    t = np.asarray(b.truth)[pool]
    runs = []
    for seed in SIM_SEEDS:
        codes, totals = ms.merged_partition(
            linkages, len(b.vectors), row_label, cut, window, judge, seed
        )
        sim = gs.simulate(codes[b.labelled][pool], t, seed=seed, open_set=OPENSET)
        runs.append({**sim, **totals})
    n = runs[0]['items']
    u1 = [(r['decisions'] + check_cost * r['checks']) / n for r in runs]
    return {
        'window': window,
        'items': n,
        'u1_with_checks': float(np.mean(u1)),
        'u1_with_checks_min_max': [float(min(u1)), float(max(u1))],
        'u1_without_checks': float(np.mean([r['u1'] for r in runs])),
        'u2': float(np.nanmean([r['u2'] for r in runs])),
        'checks': float(np.mean([r['checks'] for r in runs])),
        'accepted': float(np.mean([r['accepted'] for r in runs])),
        'false_same_accepted': float(np.mean([r['false_same_accepted'] for r in runs])),
        'clusters_mean': float(np.mean([r['clusters'] for r in runs])),
    }


def run_z3(args: argparse.Namespace) -> dict[str, Any]:
    """Judge-gated merge: window chosen on the calibration half, bar read once on the held-out half."""
    z2 = json.loads(args.z2.read_text())
    cut = float(z2['cut'])
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys = detector_keys(b.meta)
    first = calibration_mask(b, args.split_manifest)
    linkages = pt.group_linkages(keys, b.vectors @ b.vectors.T)
    base_judge = ms.Judge()
    design = {
        str(w): _merge_arm(linkages, b, first, cut, w, base_judge, CHECK_COST)
        for w in (0.0, *WINDOWS)
    }
    qualifying = {w: r for w, r in design.items() if w != '0.0' and r['u2'] >= ub.U2_BAR}
    chosen = min(qualifying, key=lambda w: qualifying[w]['u1_with_checks']) if qualifying else None
    out: dict[str, Any] = {
        'cut': cut,
        'judge': {'p_same': base_judge.p_same, 'p_false_same': base_judge.p_false_same},
        'check_cost_decisions': CHECK_COST,
        'windows': list(WINDOWS),
        'design_on_calibration_half': design,
        'chosen_window': None if chosen is None else float(chosen),
    }
    if chosen is None:
        out['verdict'] = ub.FAIL
        out['reason'] = 'no window keeps u2 >= 0.95 on the calibration half'
        return out
    w = float(chosen)
    base = _merge_arm(linkages, b, ~first, cut, 0.0, base_judge, CHECK_COST)
    merged = _merge_arm(linkages, b, ~first, cut, w, base_judge, CHECK_COST)
    bar = ADOPT_FACTOR * base['u1_with_checks']
    ok = merged['u1_with_checks'] <= bar and merged['u2'] >= ub.U2_BAR
    out['heldout_once'] = {
        'no_merge': base,
        'merged': merged,
        'bar_u1_with_checks': bar,
        'u2_bar': ub.U2_BAR,
        'fall_in_u1': 1 - merged['u1_with_checks'] / base['u1_with_checks'],
    }
    out['verdict'] = ub.PASS if ok else ub.FAIL
    out['sensitivity_after_the_read_not_barred'] = {
        f'false_same_{fs}_cost_{cost}': _merge_arm(
            linkages, b, ~first, cut, w, ms.Judge(p_false_same=fs), cost
        )
        for fs, cost in ((0.015, 1.0), (0.06, 1.0), (0.03, 0.5), (0.03, 2.0))
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    z1 = sub.add_parser('z1')
    z1.add_argument('--set', nargs=5, action='append', required=True, metavar='X')
    z1.add_argument('--out', type=Path, required=True)
    z2 = sub.add_parser('z2')
    for opt in ('bundle', 'labels', 'coco-gt', 'split-manifest', 'out'):
        z2.add_argument(f'--{opt}', type=Path, required=True)
    z3 = sub.add_parser('z3')
    for opt in ('bundle', 'labels', 'coco-gt', 'split-manifest', 'z2', 'out'):
        z3.add_argument(f'--{opt}', type=Path, required=True)
    args = ap.parse_args(argv)
    run = {'z1': run_z1, 'z2': run_z2, 'z3': run_z3}[args.cmd]
    start = os.getloadavg()
    result = {
        'command': args.cmd,
        'git_sha': scorecard.current_git_sha(),
        'load_avg_start': start,
        **run(args),
        'load_avg_end': os.getloadavg(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

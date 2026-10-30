"""Experiment runs: Y3 (chaining by size), Y2 (label-free cut rule), Y1 (fresh acceptance set), Y4.

    y_run.py y3 --bundle STEM --keys person motorcycle car truck --dendrograms DIR --out OUT.json

Set P figures are sizes and merge heights only; nothing names an item.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import eval_split, scorecard
from scripts.research.cluster_diag import (
    chaining as ch,
    class_prior as cp,
    descent as dn,
    diagnostics as dg,
    heldout as ho,
    key_cut_rule as kr,
    partitions as pt,
    scale as sc,
    y_acceptance as ya,
)
from scripts.research.cluster_diag.class_prior_run import EXTRA_SPLIT_SEEDS, stored_partition
from scripts.research.cluster_diag.dataset import detector_keys, load_bundle, read_jsonl
from scripts.research.cluster_diag.run import SEED, calibration_mask


Y3_CUTS = (0.346, 0.30, 0.25, 0.20, 0.15)


def key_dendrogram(vectors: Any, cache: Path | None) -> Any:
    """Average-linkage dendrogram of one key, read from ``cache`` when it exists."""
    if cache is not None and cache.exists():
        return np.load(cache)
    z = sc.dense_average_linkage(vectors)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, z)
    return z


def run_y3(args: argparse.Namespace) -> dict[str, Any]:
    vectors = dg.l2_normalize(np.load(args.bundle.with_suffix('.npy'), mmap_mode='r')[:])
    keys = detector_keys(read_jsonl(args.bundle.with_suffix('.meta.jsonl')))
    out: dict[str, Any] = {'cuts': list(args.cuts), 'keys': {}}
    for key in args.keys:
        rows = vectors[keys == key]
        t0 = time.perf_counter()
        z = key_dendrogram(rows, args.dendrograms / f'{key}.npy')
        out['keys'][key] = {
            'crops': len(rows),
            'fit_seconds_or_cache_read': time.perf_counter() - t0,
            'by_cut': {str(c): ch.chaining_report(z, c) for c in args.cuts},
        }
    return out


def _fold_pair(
    b: Any, keys: Any, arms: dict[str, ya.Arm], incumbent: Any, first_half: Any, frozen: Any
) -> dict[str, Any]:
    """Both folds of every arm; ``frozen`` maps an arm name to a parameter fixed beforehand."""
    ids = [m['crop_id'] for m in b.meta]
    out: dict[str, Any] = {}
    for fold, fit in (
        ('fold1_calibration_to_heldout', first_half),
        ('fold2_heldout_to_calibration', ~first_half),
    ):
        out[fold] = {
            name: ya.evaluate_fold(
                b, keys, arm, incumbent, fit, ids=ids, seed=SEED, frozen=frozen.get(name)
            )
            for name, arm in arms.items()
        }
    return out


def _common(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys = detector_keys(b.meta)
    incumbent = pt.ivf_current(b.vectors)
    stored = stored_partition(args.stored_partition, b)
    return b, keys, incumbent[b.labelled], incumbent, stored


def _set_p_rule(args: argparse.Namespace, alpha: float) -> dict[str, Any]:
    """The rule's cut on every set P key, and the chaining profile of the large keys at that cut."""
    vectors = dg.l2_normalize(np.load(args.setp_bundle.with_suffix('.npy'), mmap_mode='r')[:])
    keys = detector_keys(read_jsonl(args.setp_bundle.with_suffix('.meta.jsonl')))
    scales = kr.key_scales(vectors, keys)
    sizes = {k: int((keys == k).sum()) for k in scales}
    cuts = kr.rule_cuts(scales, alpha)
    refused = [k for k, c in cuts.items() if c is None]
    out: dict[str, Any] = {
        'alpha': alpha,
        'keys': len(scales),
        'refused_keys_of_10_crops_or_fewer': len(refused),
        'crops_in_refused_keys': sum(sizes[k] for k in refused),
        'crops': len(keys),
        'by_key': {
            k: {'crops': sizes[k], 'scale': scales[k], 'rule_cut': cuts[k]}
            for k in sorted(scales, key=lambda x: -sizes[x])
            if sizes[k] >= args.min_key_report
        },
        'chaining_at_rule_cut': {},
    }
    for k, info in out['by_key'].items():
        z = key_dendrogram(vectors[keys == k], args.dendrograms / f'{k}.npy')
        out['chaining_at_rule_cut'][k] = ch.chaining_report(z, info['rule_cut'])
    return out


def run_y2(args: argparse.Namespace) -> dict[str, Any]:
    b, keys, inc_lab, _, stored = _common(args)
    first_half = calibration_mask(b, args.split_manifest)
    arms = {'global_cut': ya.global_cut_arm(b, keys), 'rule': ya.rule_arm(b, keys)}
    folds = _fold_pair(b, keys, arms, inc_lab, first_half, {})
    op = folds['fold1_calibration_to_heldout']['rule']['operating_point']
    scales = {k: v for k, v in kr.key_scales(b.vectors, keys).items() if v is not None}
    out: dict[str, Any] = {
        'seed': SEED,
        'n_crops': len(b.vectors),
        'n_labelled': len(b.truth),
        'calibration_labelled': int(first_half.sum()),
        'folds': folds,
        'frontier_fit_on_calibration': {
            name: ya.frontier(b, arm, first_half) for name, arm in arms.items()
        },
        'rule_fit': None if op is None else {'alpha': op['parameter'], 'target': op['target']},
        'public_scales_of_keys_with_100_or_more_crops': {
            k: {'crops': int((keys == k).sum()), 'scale': v}
            for k, v in scales.items()
            if (keys == k).sum() >= 100
        },
        'public_key_scales': {
            'keys_with_a_rule_cut': len(scales),
            'min': min(scales.values()),
            'median': float(np.median(list(scales.values()))),
            'max': max(scales.values()),
        },
        'stored_partition_heldout_all': cp.views(b.truth, stored, ~first_half, b.strata_masks())[
            'all'
        ],
    }
    if op is not None:
        out['set_p'] = _set_p_rule(args, float(op['parameter']))
    return out


def _split_masks(b: Any, args: argparse.Namespace) -> dict[str, Any]:
    masks = {'pinned': calibration_mask(b, args.split_manifest)}
    pin = json.loads(args.pin.read_text())
    groups, strata, _ = eval_split.groups_from_pin(pin)
    for seed in EXTRA_SPLIT_SEEDS:
        manifest = eval_split.build_manifest(
            f'{args.pin.stem}_seed{seed}', seed, 'source_image', groups, strata
        )
        masks[f'seed_{seed}'] = ho.half_mask(
            b.group, eval_split.split_groups_of(manifest, 'calibration')
        )
    return masks


def run_y1(args: argparse.Namespace) -> dict[str, Any]:
    """Bars A1 to A7 on the fresh set. Read once: the rule's alpha comes from the Y2 run."""
    alpha = json.loads(args.rule_fit.read_text())['rule_fit']['alpha']
    b, keys, inc_lab, _, stored = _common(args)
    arms = {'global_cut': ya.global_cut_arm(b, keys), 'rule': ya.rule_arm(b, keys)}
    splits = {}
    for name, mask in _split_masks(b, args).items():
        splits[name] = {
            'calibration_labelled': int(mask.sum()),
            'heldout_labelled': int((~mask).sum()),
            **_fold_pair(b, keys, arms, inc_lab, mask, {'rule': alpha}),
        }
    pinned = calibration_mask(b, args.split_manifest)
    return {
        'seed': SEED,
        'rule_alpha_frozen_from': str(args.rule_fit.name),
        'rule_alpha': alpha,
        'n_crops': len(b.vectors),
        'n_labelled': len(b.truth),
        'detector_keys': len(set(keys.tolist())),
        'person_share_of_labelled': float((b.truth == cp.PERSON).mean()),
        'splits': splits,
        'stored_partition_heldout_all': cp.views(b.truth, stored, ~pinned, b.strata_masks())['all'],
    }


def _descent_at(b: Any, keys: Any, fit: Any, arm: ya.Arm, target: float) -> dict[str, Any]:
    chosen = ho.choose_and_score(arm.candidates, b.truth, fit, ~fit, target)
    if chosen is None:
        return {'verdict': 'INCONCLUSIVE', 'reason': f'no cut reaches pair recall {target}'}
    cut = float(chosen['parameter'])
    label_of = {int(row): b.truth[i] for i, row in enumerate(b.labelled) if fit[i]}
    rows: list[dict[str, Any]] = []
    for idx, z in pt.group_linkages(keys, b.vectors @ b.vectors.T):
        if z is None:
            continue
        local = {pos: label_of[int(r)] for pos, r in enumerate(idx) if int(r) in label_of}
        rows += dn.judge_key(z, cut, local)
    return {'cut': cut, **dn.summarise(rows)}


def run_y4(args: argparse.Namespace) -> dict[str, Any]:
    """Descent on the calibration half only. Target 0.3 is the pre-registered cut; the other
    targets are supplementary rows."""
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys = detector_keys(b.meta)
    fit = calibration_mask(b, args.split_manifest)
    arm = ya.global_cut_arm(b, keys)
    return {
        'seed': SEED,
        'calibration_labelled': int(fit.sum()),
        'by_target_pair_recall': {
            str(t): _descent_at(b, keys, fit, arm, t) for t in (0.3, 0.2, 0.1)
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    y3 = sub.add_parser('y3')
    y3.add_argument('--bundle', type=Path, required=True)
    y3.add_argument('--keys', nargs='+', required=True)
    y3.add_argument('--cuts', type=float, nargs='+', default=list(Y3_CUTS))
    y3.add_argument('--dendrograms', type=Path, required=True)
    y3.add_argument('--out', type=Path, required=True)
    y4 = sub.add_parser('y4')
    for opt in ('bundle', 'labels', 'coco-gt', 'split-manifest', 'out'):
        y4.add_argument(f'--{opt}', type=Path, required=True)
    for name in ('y2', 'y1'):
        p = sub.add_parser(name)
        for opt in ('bundle', 'labels', 'coco-gt', 'split-manifest', 'stored-partition', 'out'):
            p.add_argument(f'--{opt}', type=Path, required=True)
        if name == 'y2':
            p.add_argument('--setp-bundle', type=Path, required=True)
            p.add_argument('--dendrograms', type=Path, required=True)
            p.add_argument('--min-key-report', type=int, default=1000)
        else:
            p.add_argument('--pin', type=Path, required=True)
            p.add_argument('--rule-fit', type=Path, required=True)
    args = ap.parse_args(argv)
    run = {'y3': run_y3, 'y2': run_y2, 'y1': run_y1, 'y4': run_y4}[args.cmd]
    load_start = os.getloadavg()
    result = {
        'command': args.cmd,
        'git_sha': scorecard.current_git_sha(),
        'load_avg_start': load_start,
        **run(args),
        'load_avg_end': os.getloadavg(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

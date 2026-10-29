"""Run the class-prior experiments on an exported project (CPU only, seeded).

    class_prior_run.py <command> --bundle STEM --labels L.jsonl --coco-gt GT.json \\
        --split-manifest SPLIT.json --out OUT.json [command options]

Commands, all reading the same bundle and cutting on the calibration half, scoring on the held-out
half and then the reverse:

* ``partition``       within-detector-class average linkage against the refit IVF configuration,
                      the global cut and the trivial baselines, with the product bars
* ``splits``          the same bars on further seeded splits of the pinned evaluation set
* ``representations`` raw, whitened, k-reciprocal and DBA distances inside the class
* ``novelty``         novel-candidate clusters and cross-class merge suggestions
* ``transfer``        the cut applied to a second project, and the sample size needed to choose it
* ``streaming``       assigning late arrivals to existing clusters of their own class
* ``scale-check``     sparse against dense average linkage on this bundle
* ``scale-time``      time and peak memory of dense or sparse linkage on one key of a large bundle
"""

from __future__ import annotations

import argparse
import json
import os
import resource
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
from scripts.bench import cluster_eval_metrics as met, eval_split, scorecard
from scripts.bench.parity_metrics import adjusted_rand_index
from scripts.research.cluster_diag import (
    class_prior as cp,
    class_prior_bars as bars,
    cut_transfer as ct,
    diagnostics as dg,
    heldout as ho,
    novelty as nv,
    partitions as pt,
    representation_arms as ra,
    scale as sc,
    streaming as st,
)
from scripts.research.cluster_diag.dataset import (
    Bundle,
    detector_keys,
    image_name,
    load_bundle,
    read_jsonl,
)
from scripts.research.cluster_diag.run import SEED, calibration_mask
from scripts.research.cluster_diag.section2_reference import reproduction


EXTRA_SPLIT_SEEDS = (20261030, 20261031, 20261101)


def stored_partition(path: Path, b: Bundle) -> np.ndarray:
    """The pipeline's stored ``cluster_id`` per labelled crop, as cluster codes."""
    by_item = {r['item_id']: r['cluster_id'] for r in read_jsonl(path)}
    ids = [by_item.get(b.meta[i]['crop_id']) for i in b.labelled]
    codes: dict[Any, int] = {}
    return np.array(
        [met.UNASSIGNED if c is None else codes.setdefault(c, len(codes)) for c in ids],
        dtype=np.int64,
    )


class PartitionContext:
    """Everything the bar report needs that does not depend on which half is the calibration."""

    def __init__(self, b: Bundle, stored: np.ndarray) -> None:
        self.b = b
        self.keys = detector_keys(b.meta)
        self.stored = stored
        self.clusterer = cp.WithinKeyClusterer(b.vectors, self.keys)
        lab = b.labelled
        sim = b.vectors @ b.vectors.T
        self.families = {
            'within_detector_class': [
                (cut, self.clusterer.partition(cut)[lab]) for cut in cp.quantile_cut_grid(b.vectors)
            ],
            'global_average_ahc': [(t, p[lab]) for t, p in pt.ahc_cuts(sim, 'average')],
        }
        self.ivf_full = pt.ivf_current(b.vectors)
        self.references = {
            'ivf_refit': self.ivf_full[lab],
            **{k: v[lab] for k, v in cp.trivial_partitions(self.keys).items()},
        }

    def split_report(self, first_half: np.ndarray) -> dict[str, Any]:
        """Both folds of the bar report for one calibration/held-out division of the images."""
        b, lab = self.b, self.b.labelled
        cuts = [
            c['parameter']
            for fit in (first_half, ~first_half)
            if (
                c := ho.choose_and_score(
                    self.families['within_detector_class'], b.truth, fit, ~fit, bars.PRODUCT_TARGET
                )
            )
        ]
        ids = [m['crop_id'] for m in b.meta]
        aris = {
            str(cut): bars.permutation_ari(
                lambda v, k, cut=cut: cp.WithinKeyClusterer(v, k).partition(cut),
                b.vectors,
                self.keys,
                ids,
                SEED,
            )
            for cut in cuts
        }
        moved = {
            str(cut): met.moved_items(self.stored, self.clusterer.partition(cut)[lab], b.protected)
            for cut in cuts
        }
        folds = bars.fold_report(
            self.families,
            self.references,
            b.truth,
            b.strata_masks(),
            first_half,
            bar_family='within_detector_class',
            permutation_ari_value=min(aris.values()),
            protected_moved=max(moved.values()),
        )
        return {
            'permutation_ari_by_cut': aris,
            'protected_items_moved_by_cut': moved,
            'folds': folds,
        }


def run_partition(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    stored = stored_partition(args.stored_partition, b)
    ctx = PartitionContext(b, stored)
    report = ctx.split_report(calibration_mask(b, args.split_manifest))
    return {
        'seed': SEED,
        'split_manifest': str(args.split_manifest),
        'n_crops': len(b.vectors),
        'n_labelled': len(b.truth),
        'detector_keys': len(set(ctx.keys.tolist())),
        'protected_items': int(b.protected.sum()),
        'ivf_refit_vs_stored_ari': float(adjusted_rand_index(stored, ctx.references['ivf_refit'])),
        **report,
        'reproduction_of_decision_note_section_2': reproduction(
            report['folds']['fold1_calibration_to_heldout']
        ),
    }


def _extra_manifest(pin_path: Path, seed: int) -> dict[str, Any]:
    pin = json.loads(pin_path.read_text())
    groups, strata, _ = eval_split.groups_from_pin(pin)
    return eval_split.build_manifest(
        f'{pin_path.stem}_seed{seed}', seed, 'source_image', groups, strata
    )


def run_splits(args: argparse.Namespace) -> dict[str, Any]:
    """The product bars on the pinned split and on further seeded splits of the same pin."""
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    ctx = PartitionContext(b, stored_partition(args.stored_partition, b))
    splits = {'pinned': calibration_mask(b, args.split_manifest)}
    for seed in EXTRA_SPLIT_SEEDS:
        manifest = _extra_manifest(args.pin, seed)
        splits[f'seed_{seed}'] = ho.half_mask(
            b.group, eval_split.split_groups_of(manifest, 'calibration')
        )
    out: dict[str, Any] = {'seed': SEED, 'extra_seeds': list(EXTRA_SPLIT_SEEDS), 'splits': {}}
    for name, mask in splits.items():
        report = ctx.split_report(mask)
        cells = {}
        for fold, body in report['folds'].items():
            cell = body['families']['within_detector_class'][str(bars.PRODUCT_TARGET)]
            cells[fold] = (
                None
                if cell is None
                else {
                    'cut': cell['cut'],
                    'bars': cell['bars'],
                    'shortfalls': bars.bar_shortfalls(cell['bars']),
                    'missed_by_more_than_0.03': bars.bars_missed_by_more_than(cell['bars']),
                    'ivf_refit_bcubed': {
                        k: body['references']['ivf_refit']['all'][k]
                        for k in ('bcubed_precision', 'bcubed_recall')
                    },
                }
            )
        out['splits'][name] = {
            'calibration_labelled': int(mask.sum()),
            'heldout_labelled': int((~mask).sum()),
            'permutation_ari_by_cut': report['permutation_ari_by_cut'],
            'cells': cells,
        }
    return out


def run_representations(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    pool = dg.l2_normalize(np.load(args.fit_pool))
    return {
        'seed': SEED,
        'fit_pool': str(args.fit_pool),
        'arms': ra.run_representations(
            b, detector_keys(b.meta), calibration_mask(b, args.split_manifest), pool
        ),
    }


def _product_cut(ctx_families: Any, truth: Any, fit: np.ndarray) -> float:
    chosen = ho.choose_and_score(ctx_families, truth, fit, ~fit, bars.PRODUCT_TARGET)
    if chosen is None:
        raise SystemExit('no cut reaches the target recall on the fitting half')
    return float(chosen['parameter'])


def run_novelty(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys, lab = detector_keys(b.meta), b.labelled
    clusterer = cp.WithinKeyClusterer(b.vectors, keys)
    cands = [(c, clusterer.partition(c)[lab]) for c in cp.quantile_cut_grid(b.vectors)]
    first_half = calibration_mask(b, args.split_manifest)
    out: dict[str, Any] = {'seed': SEED, 'folds': {}}
    for name, fit in (
        ('fold1_calibration_to_heldout', first_half),
        ('fold2_heldout_to_calibration', ~first_half),
    ):
        cut = _product_cut(cands, b.truth, fit)
        out['folds'][name] = {
            'cut': cut,
            **nv.run_fold(b, keys, clusterer.partition(cut), fit, ~fit),
        }
    return out


def _grouped_labels(labels: Path, bundle: Path, out: Path) -> Path:
    """A copy of ``labels`` whose rows carry the source image file name as ``group``."""
    names = {m['crop_id']: image_name(m) for m in read_jsonl(bundle.with_suffix('.meta.jsonl'))}
    rows = [
        {**r, 'group': names[r['item_id']]} for r in read_jsonl(labels) if r['item_id'] in names
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return out


def run_transfer(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys, lab = detector_keys(b.meta), b.labelled
    clusterer = cp.WithinKeyClusterer(b.vectors, keys)
    first_half = calibration_mask(b, args.split_manifest)
    legacy = [(c, clusterer.partition(c)[lab]) for c in cp.quantile_cut_grid(b.vectors)]
    dense = [(c, clusterer.partition(c)[lab]) for c in cp.uniform_cut_grid(0.2, 0.6, 0.005)]
    cuts = {
        str(t): float(ho.choose_and_score(legacy, b.truth, first_half, ~first_half, t)['parameter'])  # type: ignore[index]
        for t in bars.TARGET_RECALLS
    }
    second_labels = _grouped_labels(
        args.second_labels, args.second_bundle, args.out.parent / 'second_labels_grouped.jsonl'
    )
    s = load_bundle(args.second_bundle, second_labels, args.second_coco_gt)
    manifest = eval_split.load_manifest(args.second_split_manifest)
    unlisted = sorted(set(s.group) - eval_split.split_groups_of(manifest, 'full'))
    held = ho.half_mask(s.group, eval_split.split_groups_of(manifest, 'heldout'))
    tuned_on = {g for g, in_cal in zip(b.group, first_half, strict=True) if in_cal}
    independent = held & np.array([g not in tuned_on for g in s.group])
    s_keys = detector_keys(s.meta)
    ivf = pt.ivf_current(s.vectors)[s.labelled]
    transfer = {}
    for t, cut in cuts.items():
        transfer[t] = {
            'on_heldout_half': ct.transfer_report(
                s.vectors, s_keys, s.truth, cut, held, s.strata_masks(), s.labelled
            ),
            'on_heldout_half_without_images_of_the_tuning_half': ct.transfer_report(
                s.vectors, s_keys, s.truth, cut, independent, s.strata_masks(), s.labelled
            ),
        }
    return {
        'seed': SEED,
        'transferred_cuts_chosen_on_calibration': cuts,
        'second_project': {
            'crops': len(s.vectors),
            'labelled': len(s.truth),
            'images_not_in_split_manifest_unscored': len(unlisted),
            'labelled_crops_not_in_split_manifest_unscored': sum(
                g in set(unlisted) for g in s.group
            ),
            'heldout_labelled': int(held.sum()),
            'heldout_labelled_not_in_tuning_half': int(independent.sum()),
            'heldout_images_in_tuning_half': len(
                {g for g, h in zip(s.group, held, strict=True) if h} & tuned_on
            ),
        },
        'transfer': transfer,
        'second_project_ivf_refit_heldout': cp.views(s.truth, ivf, held, s.strata_masks())['all'],
        'sample_size_per_key': ct.sample_size_table(
            dense,
            b.truth,
            first_half,
            bars.PRODUCT_TARGET,
            lambda eligible, n, seed: ct.subsample_per_key(keys[lab], eligible, n, seed),
            ct.PER_KEY_COUNTS,
        ),
        'sample_size_random_total': ct.sample_size_table(
            dense,
            b.truth,
            first_half,
            bars.PRODUCT_TARGET,
            ct.subsample_total,
            ct.TOTAL_COUNTS,
        ),
    }


def run_streaming(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys, lab = detector_keys(b.meta), b.labelled
    clusterer = cp.WithinKeyClusterer(b.vectors, keys)
    cands = [(c, clusterer.partition(c)[lab]) for c in cp.quantile_cut_grid(b.vectors)]
    first_half = calibration_mask(b, args.split_manifest)
    cut = _product_cut(cands, b.truth, first_half)
    manifest = eval_split.load_manifest(args.split_manifest)
    groups = nv.source_groups(b)
    runs = [
        st.run_streaming(
            b.vectors,
            keys,
            groups,
            lab,
            b.truth,
            eval_split.split_groups_of(manifest, 'calibration'),
            cut,
            seed,
        )
        for seed in st.SEEDS
    ]
    return {'seed': SEED, 'cut': cut, 'pinned_seed_run': runs[0], 'other_seed_runs': runs[1:]}


def run_scale_check(args: argparse.Namespace) -> dict[str, Any]:
    b = load_bundle(args.bundle, args.labels, args.coco_gt)
    keys, lab = detector_keys(b.meta), b.labelled
    first_half = calibration_mask(b, args.split_manifest)
    clusterer = cp.WithinKeyClusterer(b.vectors, keys)
    cands = [(c, clusterer.partition(c)[lab]) for c in cp.quantile_cut_grid(b.vectors)]
    cut = _product_cut(cands, b.truth, first_half)
    held, strata = ~first_half, b.strata_masks()
    out: dict[str, Any] = {'seed': SEED, 'cut': cut, 'rows': {}}
    dense = clusterer.partition(cut)
    sparse_runs = {}
    for k in args.neighbours:
        t0 = time.perf_counter()
        sparse = sc.within_key(
            b.vectors, keys, cut, lambda v, c, k=k: sc.sparse_average_partition(v, c, k)
        )
        sparse_runs[k] = (sparse, time.perf_counter() - t0)
    out['rows']['dense'] = cp.views(b.truth, dense[lab], held, strata)['all']
    for k, (part, secs) in sparse_runs.items():
        out['rows'][f'sparse_mutual_{k}nn'] = {
            **cp.views(b.truth, part[lab], held, strata)['all'],
            'seconds': secs,
            'ari_against_dense_all_crops': float(adjusted_rand_index(dense, part)),
        }
    dense_row = out['rows']['dense']
    for k in args.neighbours:
        row = out['rows'][f'sparse_mutual_{k}nn']
        row['bcubed_precision_gap'] = row['bcubed_precision'] - dense_row['bcubed_precision']
        row['bcubed_recall_gap'] = row['bcubed_recall'] - dense_row['bcubed_recall']
    return out


def run_scale_time(args: argparse.Namespace) -> dict[str, Any]:
    vectors = dg.l2_normalize(np.load(args.bundle.with_suffix('.npy'), mmap_mode='r')[:])
    meta = read_jsonl(args.bundle.with_suffix('.meta.jsonl'))
    keys = detector_keys(meta)
    counts = {k: int((keys == k).sum()) for k in sorted(set(keys.tolist()))}
    rows = vectors[keys == args.key]
    fit = sc.dense_average_partition if args.mode == 'dense' else sc.sparse_average_partition
    with scorecard.RunMeter(seed=SEED) as meter:
        t0 = time.perf_counter()
        part = fit(rows, args.cut)
        seconds = time.perf_counter() - t0
    sizes = np.bincount(part)
    return {
        'mode': args.mode,
        'key': args.key,
        'key_crops': len(rows),
        'cut': args.cut,
        'clusters': int(part.max()) + 1,
        'largest_cluster_share': float(sizes.max() / len(rows)),
        'fit_seconds': seconds,
        'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
        'crops_per_key': counts,
        'run_meta': meter.meta,
    }


COMMANDS = {
    'partition': (run_partition, ('stored-partition',)),
    'splits': (run_splits, ('stored-partition', 'pin')),
    'representations': (run_representations, ('fit-pool',)),
    'novelty': (run_novelty, ()),
    'transfer': (
        run_transfer,
        ('second-bundle', 'second-labels', 'second-coco-gt', 'second-split-manifest'),
    ),
    'streaming': (run_streaming, ()),
    'scale-check': (run_scale_check, ()),
}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    common = ('bundle', 'labels', 'coco-gt', 'split-manifest', 'out')
    for cmd, (_, extra) in COMMANDS.items():
        p = sub.add_parser(cmd)
        for name in (*common, *extra):
            p.add_argument(f'--{name}', type=Path, required=True)
        if cmd == 'scale-check':
            p.add_argument('--neighbours', type=int, nargs='+', default=[10, 30])
    t = sub.add_parser('scale-time')
    t.add_argument('--bundle', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--key', required=True)
    t.add_argument('--mode', choices=('dense', 'sparse'), required=True)
    t.add_argument('--cut', type=float, required=True)
    args = ap.parse_args(argv)
    run = run_scale_time if args.cmd == 'scale-time' else COMMANDS[args.cmd][0]
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

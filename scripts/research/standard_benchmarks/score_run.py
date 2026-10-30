"""Score the cached embeddings exactly as registered (``docs/research/wb2_standard_benchmarks.md``).

    WB2_CACHE=DIR python -m scripts.research.standard_benchmarks.score_run score --dataset cars196 --out RES
    WB2_CACHE=DIR python -m scripts.research.standard_benchmarks.score_run decide --out RES

``score`` writes ``RES/scores_<dataset>.json`` and ``RES/hits_<dataset>.npy`` (per-query Recall@1 hits, columns named in the JSON);
``decide`` reads them for all sets and writes ``RES/decisions.json``. Run the k-means and recall
parts on the locked GPU (``--device cuda``); the linkage is CPU.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.cluster.hierarchy import fcluster

from scripts.bench.cluster_eval_metrics import bcubed_scores, pairwise_scores
from scripts.bench.parity_metrics import adjusted_rand_index
from scripts.research.cluster_diag.diagnostics import linkage_of
from scripts.research.fine_grained.data import env_path, run_info
from scripts.research.fine_grained.metrics import choose_operating_point
from scripts.research.standard_benchmarks import bars, metrics, protocol


if TYPE_CHECKING:
    from collections.abc import Callable


KS = (1, 2, 4, 8)
SOP_KS = (1, 2, 4, 8, 10, 100, 1000)
KMEANS_SEEDS = (0, 1, 2)
N_CUTS = 60
VARIANTS = ('centre', 'letterbox')


def load_combos(cache: Path) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p) for p in sorted(cache.glob('*__*.npy'))}


def partition_scores(truth: np.ndarray, part: np.ndarray) -> dict[str, Any]:
    b = bcubed_scores(truth, part)
    pair = pairwise_scores(truth, part)
    return {
        'ari': adjusted_rand_index(truth, part),
        'nmi': metrics.nmi(truth, part),
        'bcubed_precision': b['bcubed_precision'],
        'bcubed_recall': b['bcubed_recall'],
        'bcubed_f': b['bcubed_f'],
        'pair_recall': pair['pair_recall'],
        'clusters': len(np.unique(part)),
    }


def dendrogram(vectors: np.ndarray) -> np.ndarray:
    v = vectors.astype(np.float32)
    return np.asarray(linkage_of(v @ v.T, 'average'))


def family(z: np.ndarray, truth: np.ndarray) -> list[dict[str, float]]:
    """Operating points ``{param, recall, precision}`` along the dendrogram, as WB's AHC family."""
    heights = [float(h) for h in np.quantile(z[:, 2], np.linspace(0.5, 0.995, N_CUTS))]
    pts = []
    for h in heights:
        part = fcluster(z, h, 'distance') - 1
        pts.append(
            {
                'param': h,
                'recall': pairwise_scores(truth, part)['pair_recall'] or 0.0,
                'precision': bcubed_scores(truth, part)['bcubed_precision'],
            }
        )
    return pts


def ahc_reading(
    calib_vecs: np.ndarray, calib_truth: np.ndarray, test_vecs: np.ndarray, test_truth: np.ndarray
) -> dict[str, Any]:
    """Section 3: the calibration-chosen cut applied to the test set, and the cut into K clusters."""
    t0 = time.time()
    point = choose_operating_point(family(dendrogram(calib_vecs), calib_truth))
    z = dendrogram(test_vecs)
    k = len(np.unique(test_truth))
    out: dict[str, Any] = {
        'seconds': time.time() - t0,
        'rows': len(test_truth),
        'operating_point': point,
        'cut_at_k': partition_scores(test_truth, fcluster(z, k, 'maxclust') - 1),
    }
    out['operating_cut'] = (
        partition_scores(test_truth, fcluster(z, point['param'], 'distance') - 1) if point else None
    )
    out['verdict_b3'] = bars.cut_verdict(
        point, out['operating_cut']['bcubed_precision'] if out['operating_cut'] else 0.0
    )
    return out


def timed(fn: Callable[[], object], reps: int = 3) -> list[float]:
    out = []
    for _ in range(reps):
        t0 = time.time()
        fn()
        out.append(time.time() - t0)
    return out


def kmeans_reading(vecs: np.ndarray, truth: np.ndarray, device: str) -> dict[str, Any]:
    k = len(np.unique(truth))
    nmis, secs, first = [], [], None
    for seed in KMEANS_SEEDS:
        t0 = time.time()
        assign = metrics.kmeans(vecs, k, seed, device)
        secs.append(time.time() - t0)
        nmis.append(metrics.nmi(truth, assign))
        first = assign if first is None else first
    return {
        'k': k,
        'nmi_seeds': nmis,
        'nmi_mean': float(np.mean(nmis)),
        'nmi_sd': float(np.std(nmis)),
        'seconds': secs,
        'first_assign': first,
    }


def score_dataset(args: argparse.Namespace) -> None:
    cache = env_path('WB2_CACHE') / args.dataset
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (cache / 'rows.jsonl').read_text().splitlines() if x]
    labels = np.array([r['label'] for r in rows])
    part = np.array([r['part'] for r in rows])
    test, calib = np.nonzero(part == 'test')[0], np.nonzero(part == 'calib')[0]
    spec = protocol.SETS[args.dataset]
    ks = SOP_KS if args.dataset == 'sop' else KS
    protocol_rows = [protocol.Row(r['path'], r['label'], r['part']) for r in rows]
    sub_test = np.array(protocol.cluster_subset(protocol_rows, args.dataset, 'test'))
    sub_calib = np.array(protocol.cluster_subset(protocol_rows, args.dataset, 'calib'))
    copies = np.array(protocol.file_groups(env_path('WB2_DATA_ROOT'), args.dataset, protocol_rows))[
        test
    ]
    rng = np.random.default_rng(0)
    shuffled = labels[test][rng.permutation(len(test))]
    result: dict[str, Any] = {
        **run_info(),
        'dataset': args.dataset,
        'rows': len(test),
        'expected_rows': spec.test_images,
        'classes': len(np.unique(labels[test])),
        'ks': list(ks),
        'device': args.device,
        'combos': {},
    }
    hits: dict[str, np.ndarray] = {}
    for name, vecs in load_combos(cache).items():
        t_hits = metrics.recall_hits(vecs[test], labels[test], ks, args.device)
        c_hits = metrics.recall_hits(vecs[calib], labels[calib], (1,), args.device)
        hits[name] = t_hits[:, 0]
        dedup = metrics.recall_hits(vecs[test], labels[test], ks, args.device, copies=copies)
        top1 = metrics.top1_similarity(vecs[test], args.device)
        km = kmeans_reading(vecs[test], labels[test], args.device)
        first = km.pop('first_assign')
        entry: dict[str, Any] = {
            'recall': {str(k): float(t_hits[:, j].mean()) for j, k in enumerate(ks)},
            'recall_without_identical_files': {
                str(k): float(dedup[:, j].mean()) for j, k in enumerate(ks)
            },
            'top1_similarity': {
                'quantiles': {str(q): float(np.quantile(top1, q)) for q in (0.5, 0.9, 0.99, 0.999)},
                'share_above_0.98': float((top1 > 0.98).mean()),
            },
            'calibration_recall1': float(c_hits[:, 0].mean()),
            'kmeans': km,
            'control': {
                'recall1': float(
                    metrics.recall_hits(vecs[test], shuffled, (1,), args.device)[:, 0].mean()
                ),
                'nmi': metrics.nmi(shuffled, first),
            },
        }
        if not args.skip_ahc:
            if name == f'{bars.PRODUCTION}__centre':
                entry['dendrogram_seconds'] = timed(functools.partial(dendrogram, vecs[sub_test]))
            entry['ahc'] = ahc_reading(
                vecs[sub_calib], labels[sub_calib], vecs[sub_test], labels[sub_test]
            )
        result['combos'][name] = entry
        print(args.dataset, name, entry['recall']['1'], km['nmi_mean'], flush=True)
    result['hit_columns'] = list(hits)
    np.save(out / f'hits_{args.dataset}.npy', np.stack(list(hits.values()), axis=1))
    (out / f'scores_{args.dataset}.json').write_text(json.dumps(result, indent=1))


def split_name(combo: str) -> tuple[str, str]:
    model, variant = combo.rsplit('__', 1)
    return model, variant


def best_variant(combos: dict[str, Any], model: str) -> str:
    """The variant with the higher calibration Recall@1; a tie keeps ``centre`` (the production rule)."""
    return max(
        VARIANTS, key=lambda v: (combos[f'{model}__{v}']['calibration_recall1'], v == 'centre')
    )


def decide(args: argparse.Namespace) -> None:
    out = Path(args.out)
    scores = {
        p.stem.removeprefix('scores_'): json.loads(p.read_text()) for p in out.glob('scores_*.json')
    }
    hits = {
        d: dict(zip(scores[d]['hit_columns'], np.load(out / f'hits_{d}.npy').T, strict=True))
        for d in scores
    }
    models = sorted({split_name(c)[0] for c in scores['cars196']['combos']})
    chosen = {m: best_variant(scores['cars196']['combos'], m) for m in models}

    def interval(d: str, a: str, b: str) -> dict[str, float]:
        return metrics.paired_bootstrap(hits[d][a], hits[d][b])

    dec: dict[str, Any] = {
        **run_info(),
        'selected_variant': chosen,
        'B1': {},
        'B2': {},
        'B4': bars.TEXT_TOWER,
    }
    prod = f'{bars.PRODUCTION}__{chosen[bars.PRODUCTION]}'
    others = [d for d in scores if d != 'cars196']
    for m in models:
        if m == bars.PRODUCTION:
            continue
        combo = f'{m}__{chosen[m]}'
        iv = {d: interval(d, combo, prod) for d in scores}
        dec['B1'][m] = {
            'variant': chosen[m],
            'intervals': iv,
            'verdict': bars.gain_verdict(iv['cars196'], {d: iv[d] for d in others}),
            'text_tower': bars.TEXT_TOWER[m],
        }
    for m in models:
        iv = {d: interval(d, f'{m}__letterbox', f'{m}__centre') for d in scores}
        dec['B2'][m] = {
            'intervals': iv,
            'verdict': bars.gain_verdict(iv['cars196'], {d: iv[d] for d in others}),
        }
    r1 = {m: scores['cars196']['combos'][f'{m}__{chosen[m]}']['recall']['1'] for m in models}
    dec['weak_ordering'] = bars.weak_ordering(r1)
    pair_iv = {
        (a, b): interval('cars196', f'{a}__{chosen[a]}', f'{b}__{chosen[b]}')
        for a in models
        for b in models
        if a < b
    }
    dec['set_c_contradictions'] = bars.ordering_contradictions(r1, pair_iv)
    dec['sanity'] = bars.sanity(
        {
            m: {
                'recall': scores['cars196']['combos'][f'{m}__{chosen[m]}']['recall'],
                'control_recall1': scores['cars196']['combos'][f'{m}__{chosen[m]}']['control'][
                    'recall1'
                ],
                'control_nmi': scores['cars196']['combos'][f'{m}__{chosen[m]}']['control']['nmi'],
                'nmi': scores['cars196']['combos'][f'{m}__{chosen[m]}']['kmeans']['nmi_mean'],
                'rows': scores['cars196']['rows'],
                'expected_rows': scores['cars196']['expected_rows'],
                'all_recall_max': max(
                    scores[d]['combos'][f'{m}__{v}']['recall']['1']
                    for d in scores
                    for v in VARIANTS
                ),
            }
            for m in models
        }
    )
    (out / 'decisions.json').write_text(json.dumps(dec, indent=1))
    print(
        json.dumps(
            {k: dec[k] for k in ('selected_variant', 'weak_ordering', 'set_c_contradictions')},
            indent=1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('--dataset', required=True, choices=sorted(protocol.SETS))
    s.add_argument('--out', required=True)
    s.add_argument('--device', default='cuda')
    s.add_argument('--skip-ahc', action='store_true')
    s.set_defaults(fn=score_dataset)
    d = sub.add_parser('decide')
    d.add_argument('--out', required=True)
    d.set_defaults(fn=decide)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()

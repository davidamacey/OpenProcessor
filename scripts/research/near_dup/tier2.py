"""nd:P1 tier 2 analysis: the registered bars on live vectors, and the harness-versus-live comparison.

Inputs are the outputs of ``live_ingest.py`` (live vectors and the route's stored groups) and, for the
comparison, ``encode.py`` vectors of the same files. Rule: note section 7.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.research.near_dup import sweep


def complete_families(rows: list[dict], have: set[int]) -> set[int]:
    by_fam: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_fam.setdefault(r['family'], []).append(i)
    return {f for f, idx in by_fam.items() if all(i in have for i in idx)}


def verdict(x: np.ndarray, rows: list[dict], families: set[int], cut: float) -> dict:
    keep = [i for i, r in enumerate(rows) if r['family'] in families]
    sub_rows = [rows[i] for i in keep]
    pool = sweep.Pool(x[keep], sub_rows)
    ctl = pool.evaluate('single', sweep.CONTROL_CUT, 50)
    arm = pool.evaluate('single', cut, 50)
    return {
        'cut': cut,
        'families': len(families),
        'items': len(keep),
        'passes_all_bars': sweep.passes(arm, ctl),
        'arm': sweep.trim(arm),
        'control_0.98': sweep.trim(ctl),
    }


def live_groups_vs_exact(rows: list[dict], groups: dict[int, str | None], have: set[int]) -> dict:
    """Per transform: share of copies the route grouped with their original."""
    orig = {r['family']: i for i, r in enumerate(rows) if r['transform'] == 'orig'}
    per: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r['transform'] == 'orig' or i not in have or orig[r['family']] not in have:
            continue
        g, go = groups.get(i), groups.get(orig[r['family']])
        per.setdefault(r['transform'], []).append(int(g is not None and g == go))
    return {
        t: {'recovered': sum(v), 'n': len(v), 'recall': sum(v) / len(v)} for t, v in per.items()
    }


def compare(live: np.ndarray, harness: np.ndarray, rows: list[dict], idx: list[int]) -> dict:
    per: dict[str, list[float]] = {}
    for k, i in enumerate(idx):
        per.setdefault(rows[i]['transform'], []).append(float(live[i] @ harness[k]))
    allv = [v for vs in per.values() for v in vs]
    return {
        'n': len(allv),
        'median': float(np.median(allv)),
        'min': float(np.min(allv)),
        'per_transform_median': {t: float(np.median(v)) for t, v in sorted(per.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--live', type=Path, required=True, help='live_ingest.py output prefix')
    ap.add_argument('--cut', type=float, required=True)
    ap.add_argument(
        '--harness', type=Path, help='encode.py prefix, vectors of the same rows (subset allowed)'
    )
    ap.add_argument(
        '--harness-rows', type=Path, help='text file of row indices that --harness covers'
    )
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    rows = sweep.pool_from_manifest(json.loads(args.manifest.read_text()))
    x = np.load(args.live.with_suffix('.npy'))
    meta = json.loads(args.live.with_suffix('.meta.json').read_text())
    have = set(meta['have'])
    out: dict = {'ingested_items': len(have), 'complete_families': 0}
    fams = complete_families(rows, have)
    out['complete_families'] = len(fams)
    if fams:
        out['verdict'] = verdict(x, rows, fams, args.cut)
    if 'groups' in meta:
        out['route_groups'] = live_groups_vs_exact(
            rows, {int(k): v for k, v in meta['groups'].items()}, have
        )
        out['route_meta'] = meta['ingest']
    if 'statuses' in meta:
        out['statuses'] = meta['statuses']
    if args.harness and args.harness_rows:
        idx = [int(v) for v in args.harness_rows.read_text().split()]
        h = np.load(args.harness.with_suffix('.npy'))
        pairs = [(k, i) for k, i in enumerate(idx) if i in have]
        out['live_vs_harness'] = compare(x, h[[k for k, _ in pairs]], rows, [i for _, i in pairs])
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    sys.stdout.write(
        json.dumps({k: v for k, v in out.items() if k not in ('verdict',)})[:1500] + '\n'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

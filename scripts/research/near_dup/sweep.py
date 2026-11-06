"""nd:P1: copy-level cuts per space and grouping rule on the constructed copy set (and Copydays).

Registration: ``docs/research/near_duplicate_detection.md`` section 1. The cut is chosen on the calibration
half of each split; the test half is read once, at the chosen cut and at the control cut. Subcommands:

* ``paths``: write the encode list (originals first, then every copy) for a manifest;
* ``run``: the sweep for one space (``--space``), with the vectors from ``encode.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.research.near_dup import build_sets, group


GRID = [round(0.90 + 0.005 * k, 3) for k in range(20)]
RULES = ('single', 'single_guard', 'complete')
CONTROL_CUT = 0.98
MAX_FALSE_GROUP_RATE = 0.005
RECALL_BARS = {'T-q60': 0.90, 'T-half': 0.90, 'T-crop90': 0.80}


def pool_from_manifest(manifest: dict) -> list[dict]:
    """Rows ``{path, family, transform, source}``: originals first, then copies, in manifest order."""
    rows = [
        {'path': o['path'], 'family': i, 'transform': 'orig', 'source': o['source']}
        for i, o in enumerate(manifest['originals'])
    ]
    source = {i: o['source'] for i, o in enumerate(manifest['originals'])}
    rows += [
        {
            'path': c['path'],
            'family': c['family'],
            'transform': c['transform'],
            'source': source[c['family']],
        }
        for c in manifest['copies']
    ]
    return rows


class Pool:
    def __init__(self, x: np.ndarray, rows: list[dict]) -> None:
        self.x = x
        self.family = np.array([r['family'] for r in rows], dtype=np.int64)
        self.transform = np.array([r['transform'] for r in rows])
        self.source = np.array([r['source'] for r in rows])
        self.original_index = {
            int(f): int(i)
            for i, (f, t) in enumerate(zip(self.family, self.transform, strict=True))
            if t == 'orig'
        }
        self.edges = group.edges_from_vectors(x)

    def evaluate(self, rule: str, cut: float, guard: int) -> dict:
        labels, n_split = group.group_labels(self.x, self.edges, cut, rule, guard=guard)
        out = group.evaluate(labels, self.family, self.transform, self.original_index)
        out['n_split'] = n_split
        out['edges'] = int((self.edges.cos >= cut).sum())
        return out


def subset(pool_x: np.ndarray, rows: list[dict], families: set[int]) -> Pool:
    idx = [i for i, r in enumerate(rows) if r['family'] in families]
    return Pool(pool_x[idx], [rows[i] for i in idx])


def passes(m: dict, control: dict) -> bool:
    """The registered bars (section 1.8) for one metrics dict against the control's metrics at 0.98."""
    per = m['per_transform']
    if m['false_group_rate'] > MAX_FALSE_GROUP_RATE:
        return False
    for name, bar in RECALL_BARS.items():
        if per[name]['recall'] < bar:
            return False
    return all(per[t]['recall'] >= control['per_transform'][t]['recall'] for t in per)


def choose_cut(curve: dict[float, dict], control: dict) -> float | None:
    """Highest cosine on the grid whose metrics pass every bar, else ``None`` (keep 0.98)."""
    ok = [c for c in sorted(curve, reverse=True) if passes(curve[c], control)]
    return ok[0] if ok else None


def trim(m: dict) -> dict:
    """Keep what the note quotes; drop nothing the bars read."""
    return {
        'per_transform': {
            t: {'recall': v['recall'], 'recovered': v['recovered'], 'n': v['n'], 'ci': v['ci']}
            for t, v in m['per_transform'].items()
        },
        'pooled_recall': m['pooled_recall'],
        'false_group_rate': m['false_group_rate'],
        'false_groups': m['false_groups'],
        'groups': m['groups'],
        'false_group_ci': m['false_group_ci'],
        'largest_group': m['largest_group'],
        'bcubed': m['bcubed'],
        'n_split': m.get('n_split', 0),
        'edges': m.get('edges', 0),
    }


def split_families(n_families: int, seed: int) -> tuple[set[int], set[int]]:
    perm = np.random.default_rng(seed).permutation(n_families)
    half = n_families // 2
    return set(map(int, perm[:half])), set(map(int, perm[half:]))


def run_split(x: np.ndarray, rows: list[dict], seed: int, guard: int, grid: list[float]) -> dict:
    n_fam = 1 + max(r['family'] for r in rows)
    cal_f, test_f = split_families(n_fam, seed)
    cal = subset(x, rows, cal_f)
    control_cal = cal.evaluate('single', CONTROL_CUT, guard)
    result: dict = {'seed': seed, 'calibration': {}, 'chosen': {}, 'test': {}, 'stability': {}}
    curves: dict[str, dict[float, dict]] = {}
    for rule in RULES:
        curves[rule] = {c: cal.evaluate(rule, c, guard) for c in grid}
        result['calibration'][rule] = {str(c): trim(m) for c, m in curves[rule].items()}
        result['chosen'][rule] = choose_cut(curves[rule], control_cal)
    result['product_cross_check'] = product_cross_check(cal, [CONTROL_CUT, 0.95])
    for src in sorted(set(cal.source)):
        sub_rows = [r for r in rows if r['family'] in cal_f and r['source'] == src]
        sub = Pool(*_take(x, rows, sub_rows))
        sub_control = sub.evaluate('single', CONTROL_CUT, guard)
        sub_curve = {c: sub.evaluate('single', c, guard) for c in grid}
        result['stability'][src] = choose_cut(sub_curve, sub_control)
    test = subset(x, rows, test_f)
    result['test']['control_single_0.98'] = trim(test.evaluate('single', CONTROL_CUT, guard))
    for rule in RULES:
        cut = result['chosen'][rule]
        if cut is not None:
            result['test'][f'{rule}@{cut}'] = trim(test.evaluate(rule, cut, guard))
    return result


def _take(x: np.ndarray, rows: list[dict], keep: list[dict]) -> tuple[np.ndarray, list[dict]]:
    ids = {id(r) for r in keep}
    idx = [i for i, r in enumerate(rows) if id(r) in ids]
    return x[idx], keep


def product_cross_check(pool: Pool, cuts: list[float]) -> dict[str, bool]:
    """The research ``single`` partition equals the product ``near_dup_groups`` partition at each cut."""
    out = {}
    for cut in cuts:
        labels, _ = group.group_labels(pool.x, pool.edges, cut, 'single')
        mine = {
            frozenset(np.flatnonzero(labels == g).tolist())
            for g in np.unique(labels[np.bincount(labels)[labels] >= 2])
        }
        theirs = {frozenset(g) for g in group.product_groups(pool.x, cut)}
        out[str(cut)] = mine == theirs
    return out


def run_copydays(x: np.ndarray, rows: list[dict], guard: int, grid: list[float]) -> dict:
    rows = [{**r, 'source': 'copydays'} for r in rows]
    pool = Pool(x, rows)
    return {rule: {str(c): trim(pool.evaluate(rule, c, guard)) for c in grid} for rule in RULES}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('paths', help='write the encode path list for a manifest')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument(
        '--limit-originals', type=int, default=0, help='tier-0 smoke: first N families only'
    )
    r = sub.add_parser('run', help='the sweep for one space')
    r.add_argument('--manifest', type=Path, required=True)
    r.add_argument('--vectors', type=Path, required=True, help='encode.py output prefix')
    r.add_argument('--space', required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--splits', type=int, nargs='*', default=[1, 2, 3])
    r.add_argument('--guard', type=int, default=50)
    r.add_argument('--registration-commit', required=True)
    r.add_argument('--code-commit', required=True, help='git rev-parse HEAD of the code that ran')
    r.add_argument('--limit-originals', type=int, default=0)
    r.add_argument('--copydays-vectors', type=Path, help='encode.py prefix of the Copydays items')
    r.add_argument('--copydays-root', type=Path, help='Copydays root (original/, strong/, ...)')
    args = ap.parse_args(argv)

    if args.cmd == 'paths' and args.copydays_root:
        cd = build_sets.read_copydays(args.copydays_root)
        args.out.write_text('\n'.join(str(r['path']) for r in cd) + '\n')
        sys.stdout.write(f'{len(cd)} copydays paths\n')
        return 0
    manifest = json.loads(args.manifest.read_text())
    if args.limit_originals:
        keep = args.limit_originals
        manifest['originals'] = manifest['originals'][:keep]
        manifest['copies'] = [c for c in manifest['copies'] if c['family'] < keep]
    rows = pool_from_manifest(manifest)
    if args.cmd == 'paths':
        args.out.write_text('\n'.join(r['path'] for r in rows) + '\n')
        sys.stdout.write(f'{len(rows)} paths\n')
        return 0
    x = np.load(args.vectors.with_suffix('.npy'))
    meta = json.loads(args.vectors.with_suffix('.json').read_text())
    if meta['paths'] != [r['path'] for r in rows]:
        raise SystemExit('vector order does not match the manifest pool')
    result: dict = {
        'space': args.space,
        'registration_commit': args.registration_commit,
        'code_commit': args.code_commit,
        'guard': args.guard,
        'grid': GRID,
        'n_items': len(rows),
        'n_families': 1 + max(r['family'] for r in rows),
        'splits': [run_split(x, rows, s, args.guard, GRID) for s in args.splits],
    }
    if args.copydays_vectors and args.copydays_root:
        cx = np.load(args.copydays_vectors.with_suffix('.npy'))
        result['copydays'] = run_copydays(
            cx, json.loads(args.copydays_rows.read_text()), args.guard, GRID
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    sys.stdout.write(f'wrote {args.out}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

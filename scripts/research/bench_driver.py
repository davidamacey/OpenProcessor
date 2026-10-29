"""Run prototype methods as separate processes, score them with cluster_eval.py, tabulate.

    bench_driver.py --python /path/to/venv/bin/python --vectors v.npz --labels labels.jsonl \\
        --work-dir out/ --baseline ivf_ahc --run leiden_init:leiden_init:--gamma=0.01 \\
        --run hdb:hdbscan_leaf [--seeds 0 1] [--timeout-s 3600] [--tag 20k]

Each ``--run NAME:METHOD[:ARG,ARG...]`` is one process per seed (peak RSS is that process's
alone); the baseline partition is produced first and passed as ``--partition`` to every run.
Seed 0 is scored against the labels with seed 1 as the second partition, so stability is the ARI
of two runs. Results go to ``<work-dir>/<tag>_results.json`` and print as a markdown table.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COLS = (
    'pair_precision',
    'pair_recall',
    'false_merge_rate',
    'false_split_rate',
    'correct_split_recovery',
    'unassigned_fraction',
    'clusters',
)


MEM_LIMIT_GB = 36.0


def sh(
    cmd: list[str], timeout: float | None, mem_gb: float = 0.0
) -> subprocess.CompletedProcess[str]:
    cap = int(mem_gb * 2**30)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        preexec_fn=(lambda: resource.setrlimit(resource.RLIMIT_AS, (cap, cap))) if cap else None,
    )  # nosec B603  argv list built from our own arguments, no shell


def one_run(
    args: argparse.Namespace, name: str, method: str, extra: list[str], seed: int, base: Path | None
) -> dict[str, Any]:
    out = Path(args.work_dir) / f'{args.tag}_{name}_s{seed}.jsonl'
    cmd = [
        args.python,
        str(HERE / 'leiden_refine_proto.py'),
        'run',
        '--vectors',
        args.vectors,
        '--method',
        method,
        '--out',
        str(out),
        '--seed',
        str(seed),
        *extra,
    ]
    if base:
        cmd += ['--partition', str(base)]
    try:
        # an address-space cap breaks CUDA (cuSPARSE fails with a kernel launch error), so GPU methods run uncapped
        res = sh(cmd, args.timeout_s, 0.0 if method.startswith('gpu_') else args.mem_limit_gb)
    except subprocess.TimeoutExpired:
        return {'name': name, 'seed': seed, 'error': f'timeout {args.timeout_s}s'}
    if res.returncode:
        return {'name': name, 'seed': seed, 'error': res.stderr.strip().splitlines()[-1:]}
    return {'name': name, 'seed': seed, 'out': str(out), **_stats(out)}


def _stats(out: Path) -> dict[str, Any]:
    stats: dict[str, Any] = json.loads(out.with_suffix('.stats.json').read_text())
    return stats


def score(args: argparse.Namespace, first: dict[str, Any], second: dict[str, Any] | None) -> dict:
    dest = Path(first['out']).with_suffix('.score.json')
    cmd = [
        args.python,
        str(REPO / 'scripts/bench/cluster_eval.py'),
        'score',
        '--labels',
        args.labels,
        '--partition',
        first['out'],
        '--out',
        str(dest),
        '--runtime-s',
        str(first['wall_s']),
        '--peak-rss-mb',
        str(first['peak_rss_mb']),
    ]
    if second and 'out' in second:
        cmd += ['--second-partition', second['out']]
    res = sh(cmd, None)
    if res.returncode:
        raise SystemExit(res.stderr)
    scored: dict[str, Any] = json.loads(dest.read_text())
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--vectors', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--baseline', default='ivf_ahc')
    ap.add_argument('--run', action='append', default=[])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--timeout-s', type=float, default=7200)
    ap.add_argument('--mem-limit-gb', type=float, default=MEM_LIMIT_GB, help='address-space cap')
    args = ap.parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    base_run = one_run(args, 'baseline', args.baseline, [], 0, None)
    if 'error' in base_run:
        raise SystemExit(f'baseline failed: {base_run}')
    base = Path(base_run['out'])
    specs: list[tuple[str, str, list[str]]] = [('baseline', args.baseline, [])]
    for spec in args.run:
        name, method, *rest = spec.split(':')
        specs.append((name, method, rest[0].split(',') if rest else []))

    rows = []
    for name, method, extra in specs:
        runs = (
            [base_run, one_run(args, 'baseline2', method, extra, 1, None)]
            if name == 'baseline'
            else [one_run(args, name, method, extra, s, base) for s in args.seeds]
        )
        if 'error' in runs[0]:
            rows.append({'name': name, 'error': runs[0]['error']})
            continue
        sc = score(args, runs[0], runs[1] if len(runs) > 1 else None)
        rows.append(
            {
                'name': name,
                'method': method,
                'args': extra,
                'wall_s': runs[0]['wall_s'],
                'stages': runs[0]['stages'],
                'peak_rss_mb': runs[0]['peak_rss_mb'],
                'gpu_peak_mb': runs[0].get('gpu_peak_mb'),
                'stability_ari': (sc.get('stability') or {}).get('ari'),
                **{c: sc.get(c) for c in COLS},
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    Path(args.work_dir, f'{args.tag}_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

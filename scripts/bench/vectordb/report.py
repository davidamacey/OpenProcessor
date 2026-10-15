"""Render the result JSON documents as the markdown tables of docs/PERFORMANCE.md.

    report.py DIR_OF_RESULTS SIZE [tables ...]     tables: write query filters ops
    report.py DIR_OF_RESULTS scale 2000,10000,100000 ARM [ARM ...]

Reads every ``<arm>_r<N>.json`` under ``DIR/SIZE``. With several repetitions a cell shows the
median and the min-max range; a difference between two arms is only claimed when the ranges
do not overlap (``vdb_lib.differs_beyond_noise``).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.vectordb.vdb_lib import median_range


def load_runs(root: Path, size: int) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((root / str(size)).glob('*_r[0-9]*.json')):
        if '_load1_' in path.name:
            continue
        arm = re.sub(r'_r\d+$', '', path.stem)
        runs.setdefault(arm, []).append(json.loads(path.read_text()))
    return runs


def load_single_client(root: Path, size: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((root / str(size)).glob('*_load1_r[0-9]*.json')):
        out.setdefault(path.name.split('_load1_')[0], []).append(json.loads(path.read_text()))
    return out


def cell(values: list[float], digits: int = 1) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return 'n/a'
    med, lo, hi = median_range(vals)
    if len(vals) == 1:
        return f'{med:.{digits}f}'
    return f'{med:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})'


def detail_row(doc: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in doc.get('query', {}).get('detail', []):
        if row['filter'] == name:
            found: dict[str, Any] = row
            return found
    return None


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(lines)


def write_table(
    runs: dict[str, list[dict[str, Any]]], single: dict[str, list[dict[str, Any]]]
) -> str:
    rows = []
    for arm, docs in runs.items():
        n = docs[0]['size']
        load = [d['load'] for d in docs]
        foot = [d['footprint'] for d in docs if 'footprint' in d]
        disk = [
            (f['data_dir_bytes'] - ld['data_dir_before_bytes']) / n
            for f, ld in zip(foot, load, strict=False)
            if f.get('data_dir_bytes') and ld.get('data_dir_before_bytes') is not None
        ]
        reported = [
            f['engine_reported_bytes'] / n for f in foot if f.get('engine_reported_bytes', 0) > 0
        ]
        anon = [
            (
                f['cgroup_anon_bytes']
                + f.get('cgroup_shmem_bytes', 0)
                - ld['cgroup_anon_before_bytes']
                - (ld.get('cgroup_shmem_before_bytes') or 0)
            )
            / n
            for f, ld in zip(foot, load, strict=False)
            if f.get('cgroup_anon_bytes')
            and ld.get('cgroup_anon_before_bytes') is not None
            and (docs[0]['config']['store'] != 'pgvector' or 'cgroup_shmem_bytes' in f)
        ]
        one = [d['load']['vectors_per_s'] for d in single.get(arm, [])]
        rows.append(
            [
                arm,
                cell([x['vectors_per_s'] for x in load], 0),
                cell(one, 0),
                cell([x['searchable_s'] for x in load], 0),
                cell([n / x['searchable_s'] for x in load], 0),
                cell([x / 1024 for x in disk], 2),
                cell([x / 1024 for x in reported], 2),
                cell([x / 1024 for x in anon], 2),
                str(len(docs)),
            ]
        )
    return table(
        [
            'arm',
            'load vec/s, 8 clients',
            'load vec/s, 1 client',
            'to searchable s',
            'vec/s incl. index build',
            'disk KiB/vec (data dir)',
            'disk KiB/vec (engine)',
            'RAM delta KiB/vec',
            'runs',
        ],
        rows,
    )


def query_table(runs: dict[str, list[dict[str, Any]]], flt: str = 'review_default') -> str:
    rows = []
    for arm, docs in runs.items():
        det = [r for r in (detail_row(d, flt) for d in docs) if r]
        if not det:
            continue
        first = det[0]
        rows.append(
            [
                arm,
                json.dumps(first['params']),
                'yes' if all(r['reached_target'] for r in det) else 'NO',
                cell([r['recall@10'] for r in det], 3),
                cell([r['p50_ms'][0] for r in det], 1),
                cell([r['p95_ms'][0] for r in det], 1),
                cell([r['qps_c8'][0] for r in det if 'qps_c8' in r], 0),
                cell([r['qps_c32'][0] for r in det if 'qps_c32' in r], 0),
                cell([r['recall@100'] for r in det], 3),
                json.dumps(first['k100_params']),
            ]
        )
    return table(
        [
            'arm',
            'setting (cheapest with recall@10 >= 0.95)',
            'reached',
            'recall@10',
            'p50 ms',
            'p95 ms',
            'QPS c8',
            'QPS c32',
            'recall@100',
            'setting for k=100',
        ],
        rows,
    )


def filter_table(runs: dict[str, list[dict[str, Any]]], arms: list[str], metric: str) -> str:
    """One row per filter, one column per arm; ``metric`` is ``recall`` (at 10) or ``p95``."""
    first = next(iter(runs.values()))[0]
    names = [r['filter'] for r in first['query']['detail']]
    rows = []
    for name in names:
        sel = detail_row(first, name)
        assert sel is not None  # names come from first's own detail rows
        row = [name, f'{100 * sel["selectivity"]:.2f}%']
        for arm in arms:
            det = [r for r in (detail_row(d, name) for d in runs.get(arm, [])) if r]
            if not det:
                row.append('n/a')
            elif metric == 'recall':
                row.append(cell([r['recall@10'] for r in det], 3))
            else:
                row.append(cell([r['p95_ms'][0] for r in det], 1))
        rows.append(row)
    return table(['filter', 'selectivity', *arms], rows)


def ops_table(runs: dict[str, list[dict[str, Any]]]) -> str:
    rows = []
    for arm, docs in runs.items():
        rew = [d['rewrite'] for d in docs if d.get('rewrite')]
        stream = [d['stream'] for d in docs if d.get('stream')]
        dele = [d['delete'] for d in docs if d.get('delete')]
        if not (rew or stream or dele):
            continue
        full = [r[-1] for r in rew]
        small = [r[0] for r in rew if len(r) > 1]
        rows.append(
            [
                arm,
                cell([r['seconds'] for r in small], 1),
                cell([r['seconds'] for r in full], 1),
                cell(
                    [
                        r['queries_during']['recall@10']
                        for r in full
                        if r['queries_during'].get('queries')
                    ],
                    3,
                ),
                cell(
                    [
                        r['queries_during']['p95_ms']
                        for r in full
                        if r['queries_during'].get('queries')
                    ],
                    1,
                ),
                cell([s['achieved_per_s'] for s in stream], 0),
                cell([s['p95_ratio'] for s in stream], 2),
                cell([x['delete_s'] for x in dele], 1),
                cell([x['bytes_before'] / 2**20 for x in dele], 0),
                cell([x['bytes_after_settle'] / 2**20 for x in dele], 0),
            ]
        )
    return table(
        [
            'arm',
            'cluster_id rewrite 12.8k s',
            'rewrite all s',
            'recall@10 during',
            'p95 ms during',
            'stream insert vec/s achieved',
            'p95 ratio mixed/idle',
            'delete 10% s',
            'MiB before delete',
            'MiB after settle',
        ],
        rows,
    )


def scale_table(root: Path, sizes: list[int], arms: list[str]) -> str:
    """Review-default p95 (ms) and recall@10, QPS at 8 clients and disk per vector, per size."""
    rows = []
    for arm in arms:
        row = [arm]
        for size in sizes:
            docs = load_runs(root, size).get(arm, [])
            det = [r for r in (detail_row(d, 'review_default') for d in docs) if r]
            if not det:
                row.append('n/a')
                continue
            foot = [d['footprint'] for d in docs if d.get('footprint')]
            disk = [
                f.get('engine_reported_bytes_per_vector') or f.get('data_dir_bytes_per_vector', 0)
                for f in foot
            ]
            row.append(
                f'{cell([r["recall@10"] for r in det], 3)} / {cell([r["p95_ms"][0] for r in det], 1)} ms / '
                f'{cell([r["qps_c8"][0] for r in det], 0)} QPS / {cell([x / 1024 for x in disk], 1)} KiB'
            )
        rows.append(row)
    return table(['arm', *[f'{n:,} vectors' for n in sizes]], rows)


def main() -> None:
    root = Path(sys.argv[1])
    if sys.argv[2] == 'scale':
        print(scale_table(root, [int(x) for x in sys.argv[3].split(',')], sys.argv[4:]))
        return
    size = int(sys.argv[2])
    which = sys.argv[3:] or ['write', 'query', 'ops']
    runs, single = load_runs(root, size), load_single_client(root, size)
    if 'write' in which:
        print(write_table(runs, single), end='\n\n')
    if 'query' in which:
        print(query_table(runs), end='\n\n')
    if 'ops' in which:
        print(ops_table(runs), end='\n\n')
    if 'filters' in which:
        arms = [a for a in sys.argv[3:] if a in runs]
        print(filter_table(runs, arms or list(runs)[:4], 'recall'), end='\n\n')
        print(filter_table(runs, arms or list(runs)[:4], 'p95'), end='\n\n')


if __name__ == '__main__':
    main()

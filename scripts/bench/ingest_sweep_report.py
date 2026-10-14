"""Render the ingest sweep JSON documents as the markdown tables recorded in docs/PERFORMANCE.md.

    ingest_sweep_report.py sweep RESULT.json --gpu 2          # concurrency / Triton arm grid
    ingest_sweep_report.py stages RESULT.json                 # per-endpoint stage cost
    ingest_sweep_report.py attrib RESULT.json                 # py-spy attribution
    ingest_sweep_report.py opensearch RESULT.json             # write ceiling arms

Reads only documents written by ``ingest_sweep_suite.py`` and ``os_write_ceiling.py``; medians and
min-max ranges come from the per-run values in the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.os_write_ceiling import ARMS as ARM_ORDER_MAP
from scripts.bench.suite_lib import summarize
from scripts.bench.suite_report import fmt, table
from src.config.settings import TritonModelConfig


ARM_ORDER = list(ARM_ORDER_MAP)


def point_key(point: str) -> tuple[int, int]:
    batch, _, conc = point.removeprefix('b').partition('_c')
    return int(batch), int(conc)


def sweep_table(doc: dict[str, Any], gpu: str, mode: str = 'upload') -> str:
    rows = []
    for arm, a in doc['sweep']['arms'].items():
        for point, modes in sorted(a['points'].items(), key=lambda kv: point_key(kv[0])):
            if mode not in modes:
                continue
            s = modes[mode]['summary']
            runs = modes[mode]['runs']
            api_cores = [r['flat']['api_cpu_s_per_image'] * r['flat']['images_per_s'] for r in runs]
            rows.append(
                [
                    arm,
                    point.replace('_c', ' x ').removeprefix('b'),
                    fmt(s['images_per_s'], 2),
                    fmt(s.get('request_p50_s'), 1),
                    fmt(s.get(f'gpu{gpu}_util_mean_pct'), 0),
                    fmt(s.get('busy_total'), 2),
                    fmt(s.get('api_cpu_s_per_image'), 3),
                    f'{sorted(api_cores)[len(api_cores) // 2]:.1f}',
                    fmt(s.get('host_busy_pct'), 0),
                    fmt(s.get(f'gpu{gpu}_sm_mhz_mean'), 0),
                    fmt(s.get(f'triton_{TritonModelConfig.PE_IMAGE_MODEL}_mean_batch'), 1),
                    fmt(s.get('load_avg_1m'), 0),
                ]
            )
    header = [
        'Triton arm',
        'batch x clients',
        'images/s',
        'request p50 s',
        f'GPU {gpu} util %',
        'model exec / wall',
        'API CPU s/image',
        'API cores',
        'host busy %',
        'SM MHz',
        'PE mean batch',
        'load avg',
    ]
    if len(doc['sweep']['arms']) == 1:  # a single arm needs no arm column
        header, rows = header[1:], [row[1:] for row in rows]
    return table(header, rows)


def stages_table(doc: dict[str, Any], arm: str = '') -> str:
    rows = []
    st = doc['stages']['arms'][arm] if arm else doc['stages']
    for name, by_threads in st.items():
        if not isinstance(by_threads, dict):
            continue
        for threads, v in by_threads.items():
            s = v['summary']
            models = sorted(
                {
                    k.removeprefix('triton_').removesuffix('_infer_ms_per_image')
                    for k in s
                    if k.endswith('_infer_ms_per_image')
                }
            )
            gpu_ms = ', '.join(
                f'{m} {s[f"triton_{m}_infer_ms_per_image"]["median"]:.1f}' for m in models
            )
            rows.append(
                [
                    name,
                    threads.removeprefix('threads_'),
                    fmt(s['images_per_s'], 1),
                    fmt(s['p50_ms'], 0),
                    fmt(s['api_cpu_s_per_image'], 3),
                    fmt(s['triton_cpu_s_per_image'], 3),
                    gpu_ms or 'n/a',
                ]
            )
    return table(
        [
            'Endpoint',
            'clients',
            'images/s',
            'p50 ms',
            'API CPU s/image',
            'Triton CPU s/image',
            'Triton exec ms/image by model',
        ],
        rows,
    )


def process_line(census: dict[str, Any]) -> str:
    """One sentence: process-level cores against the cores the surviving threads explain."""
    p, seconds = census['processes'], census['seconds']
    process = (p['user_s'] + p['system_s']) / seconds
    threads = sum(v['user_s'] + v['system_s'] for v in census['threads'].values()) / seconds
    fresh = (p['new_user_s'] + p['new_system_s']) / seconds
    return (
        f'Over {seconds:.0f} s: {process:.2f} cores in the {p["count"]} processes present throughout '
        f'({p["system_s"] / max(p["user_s"] + p["system_s"], 1e-9) * 100:.0f} % system), '
        f'{threads:.2f} of them in threads that lived the whole window, and '
        f'{fresh:.2f} cores of lifetime CPU in {p["new_count"]} processes started during it.'
    )


def thread_table(threads: dict[str, dict[str, float]], seconds: float) -> str:
    """CPU cores by thread name over the census window, largest first."""
    total = sum(v['user_s'] + v['system_s'] for v in threads.values()) or 1.0
    rows = [
        [
            f'`{name}`',
            f'{int(v["threads"])}',
            f'{(v["user_s"] + v["system_s"]) / seconds:.2f}',
            f'{v["system_s"] / max(v["user_s"] + v["system_s"], 1e-9) * 100:.0f}',
            f'{(v["user_s"] + v["system_s"]) / total * 100:.1f}',
        ]
        for name, v in sorted(
            threads.items(), key=lambda kv: -(kv[1]['user_s'] + kv[1]['system_s'])
        )[:12]
        if (v['user_s'] + v['system_s']) / seconds >= 0.005
    ]
    return table(['Thread name', 'threads', 'cores', 'system %', 'share %'], rows)


def attrib_table(doc: dict[str, Any]) -> str:
    out = []
    for mode, a in doc['attrib'].items():
        if not isinstance(a, dict) or 'profile' not in a:
            continue
        p = a['profile']
        out.append(f'`{mode}`: {p["samples"]} samples\n')
        out.append(
            table(
                ['Where the Python-side CPU is (leaf frame bucket)', 'share %'],
                [[k, f'{v * 100:.1f}'] for k, v in p['leaf_bucket_share'].items()],
            )
        )
        out.append('')
        census = a['thread_cpu_by_name']
        out.append(process_line(census))
        out.append('')
        out.append(thread_table(census['threads'], census['seconds']))
        out.append('')
        out.append(
            table(
                ['Top inclusive frames', 'share %'],
                [[f'`{f}`', f'{v * 100:.1f}'] for f, v in p['top_inclusive'][:20]],
            )
        )
        out.append('')
    return '\n'.join(out)


def opensearch_table(doc: dict[str, Any]) -> str:
    rows = []
    for arm, a in sorted(doc['arms'].items(), key=lambda kv: ARM_ORDER.index(kv[0])):
        s, c = a['summary'], a['config']
        cpu_per_k = summarize(
            [
                r['flat']['node_cpu_cores_during_load'] * 1000 / r['flat']['docs_per_s']
                for r in a['runs']
            ]
        )
        rows.append(
            [
                arm,
                f'{c["bulk"]} x {c["writers"]}',
                f'{c["refresh"]}, ef {c["ef"] or "shipped"}, {c["vectors"]}',
                fmt(s['docs_per_s'], 0),
                fmt(s['images_per_s_equivalent'], 0),
                fmt(s['searchable_s'], 0),
                fmt(s['bulk_p95_s'], 2),
                fmt(s['bytes_per_doc'], 0),
                fmt(s['node_cpu_cores_during_load'], 1),
                fmt(cpu_per_k, 2),
                fmt(s['retries_429'], 0),
            ]
        )
    return table(
        [
            'Arm',
            'bulk x writers',
            'refresh / ef / vectors',
            'docs/s',
            'images/s equiv.',
            'searchable s',
            'bulk p95 s',
            'bytes/doc',
            'node cores',
            'node CPU-s per 1k docs',
            '429 retries',
        ],
        rows,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('section', choices=('sweep', 'stages', 'attrib', 'opensearch'))
    p.add_argument('result', type=Path)
    p.add_argument('--gpu', default='2')
    p.add_argument('--mode', default='upload')
    p.add_argument('--arm', default='', help='Triton arm of a stages document run with arms')
    args = p.parse_args(argv)
    doc = json.loads(args.result.read_text())
    if args.section == 'sweep':
        print(sweep_table(doc, args.gpu, args.mode))
    elif args.section == 'stages':
        print(stages_table(doc, args.arm))
    elif args.section == 'attrib':
        print(attrib_table(doc))
    else:
        print(opensearch_table(doc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

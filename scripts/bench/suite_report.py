"""Render the baseline suite JSON as the markdown tables recorded in docs/PERFORMANCE.md.

    suite_report.py RESULT.json            # every table
    suite_report.py RESULT.json --section ingest

Reads only the document written by ``baseline_suite.py``; nothing is recomputed from
the stack. Medians and the min-max range come from the per-run values in the file.
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
from scripts.bench.suite_lib import summarize


# Numbers from docs/design/storage_sizing_and_ingest_baselines.md section 1 and 3.
STORAGE_REFERENCE = {'bytes_per_vector': 8350.0, 'bytes_per_image': 22500.0}


def fmt(summary: dict[str, Any] | None, digits: int = 1, scale: float = 1.0) -> str:
    """``median (min-max)`` for one :func:`summarize` entry."""
    if not summary or not summary.get('n'):
        return 'n/a'
    return (
        f'{summary["median"] * scale:.{digits}f} '
        f'({summary["min"] * scale:.{digits}f}-{summary["max"] * scale:.{digits}f})'
    )


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        '| ' + ' | '.join(header) + ' |',
        '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header))) + '|',
    ]
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(lines)


def environment(doc: dict[str, Any]) -> str:
    env = doc['env']
    gpu, sw = env['hardware']['gpu'], env['software']
    before = env.get('foreign_gpu_before', {})
    rows = [
        [
            'GPU (index ' + gpu['index'] + ')',
            f'{gpu["model"]}, driver {gpu["driver"]}, CUDA {gpu["cuda"]}',
        ],
        [
            'CPU',
            f'{env["hardware"]["cpu"]["model"]}, {env["hardware"]["cpu"]["logical_cores"]} logical cores',
        ],
        ['RAM', f'{env["hardware"]["ram_gb"]:.0f} GB'],
        [
            'Stack',
            f'OpenProcessor {sw["stack_version"]}, Triton {sw["triton_server"]}, OpenSearch {sw["opensearch"]}, {sw["tensorrt_runtime_lib"]}',
        ],
        ['Harness commit', sw['harness_git_sha'][:8]],
        [
            'Dataset',
            f'{env["dataset"]["count"]} images, {env["dataset"]["bytes"] / 1e6:.0f} MB, seed {env["dataset"]["seed"]}, manifest {env["dataset"]["manifest_sha256"][:16]}',
        ],
        [
            'Foreign GPU 0 load before the runs',
            f'{before.get("foreign_sm_mean_pct", 0):.1f} % SM mean over {before.get("samples", 0)} samples',
        ],
        ['Started', env['timestamp']],
    ]
    return table(['Item', 'Value'], rows)


def ingest_headline(doc: dict[str, Any]) -> str:
    rows = []
    for mode in ('upload', 'batch'):
        block = doc['ingest'].get(mode)
        if not block:
            continue
        s = block['summary']
        rows.append(
            [
                f'`/ingest/{mode}`',
                fmt(s['images_per_s'], 2),
                fmt(s['wall_s'], 0),
                fmt(s['items_per_image'], 2),
                fmt(s['request_p50_s'], 1),
                fmt(s['api_cpu_s_per_image'], 3),
                fmt(s['triton_cpu_s_per_image'], 3),
                fmt(s['cluster_train_wall_s'], 0),
                f'{s["failed"]["max"]:.0f}',
            ]
        )
    return table(
        [
            'Route',
            'images/s',
            'wall s',
            'items/image',
            'request p50 s',
            'API CPU s/image',
            'Triton CPU s/image',
            'cluster training s',
            'failed',
        ],
        rows,
    )


def ingest_stages(doc: dict[str, Any], mode: str = 'upload') -> str:
    runs = doc['ingest'][mode]['runs']
    images = doc['ingest']['config']['images']
    names = sorted({k for r in runs for k in r['stages']})
    rows = []
    for name in names:
        per_image = [r['stages'].get(name, {}).get('seconds', 0.0) / images * 1000 for r in runs]
        mean = [r['stages'].get(name, {}).get('mean_ms', 0.0) for r in runs]
        calls = [r['stages'].get(name, {}).get('calls', 0.0) / images for r in runs]
        sent = [r['stages'].get(name, {}).get('bytes', 0.0) / images for r in runs]
        rows.append(
            [
                name,
                fmt(summarize(calls), 2),
                fmt(summarize(mean), 1),
                fmt(summarize(per_image), 1),
                fmt(summarize(sent), 0, 1 / 1024) if any(sent) else 'n/a',
            ]
        )
    return table(['Stage', 'calls/image', 'mean ms/call', 'summed ms/image', 'KiB/image'], rows)


def triton_models(doc: dict[str, Any], mode: str = 'upload') -> str:
    runs = doc['ingest'][mode]['runs']
    images = doc['ingest']['config']['images']
    models = sorted({m for r in runs for m in r['triton']})
    rows = []
    for model in models:
        got = [r['triton'][model] for r in runs if model in r['triton']]
        wire = [r['triton_wire_bytes'].get(model, {}) for r in runs if model in r['triton']]
        rows.append(
            [
                f'`{model}`',
                fmt(summarize([g['inferences'] / images for g in got]), 2),
                fmt(summarize([g['mean_batch'] for g in got]), 1),
                fmt(summarize([g['queue_ms_per_request'] for g in got]), 0),
                fmt(summarize([g['exec_input_ms'] for g in got]), 1),
                fmt(summarize([g['exec_infer_ms'] for g in got]), 1),
                fmt(summarize([g['exec_output_ms'] for g in got]), 1),
                fmt(summarize([g['exec_infer_ms'] * g['executions'] / images for g in got]), 1),
                fmt(summarize([w.get('request_bytes', 0) / images for w in wire]), 0, 1 / 1024),
            ]
        )
    return table(
        [
            'Model',
            'inferences/image',
            'mean batch',
            'queue ms/request',
            'exec input ms',
            'exec infer ms',
            'exec output ms',
            'infer ms/image',
            'request KiB/image',
        ],
        rows,
    )


def gpu_rows(doc: dict[str, Any], mode: str = 'upload', gpu: str = '0') -> str:
    runs = doc['ingest'][mode]['runs']
    util = summarize([r['gpu'][gpu]['util_mean_pct'] for r in runs])
    peak = summarize([r['gpu'][gpu]['mem_peak_mb'] for r in runs])
    own = summarize([r['attribution'].get('own_sm_mean_pct', 0.0) for r in runs])
    foreign = summarize([r['attribution'].get('foreign_sm_mean_pct', 0.0) for r in runs])
    return table(
        [
            'Ingest mode',
            'GPU util mean % (nvidia-smi)',
            'GPU memory peak MiB',
            'own SM % (pmon)',
            'foreign SM % (pmon)',
        ],
        [[f'`{mode}`', fmt(util, 0), fmt(peak, 0), fmt(own, 0), fmt(foreign, 1)]],
    )


def storage(doc: dict[str, Any]) -> str:
    rows = []
    for mode in ('upload', 'batch'):
        block = doc['ingest'].get(mode)
        if not block:
            continue
        s = block['summary']
        rows.append(
            [
                f'`{mode}`',
                fmt(s['bytes_per_image'], 0),
                fmt(s['bytes_per_vector'], 0),
                f'{s["bytes_per_vector"]["median"] / STORAGE_REFERENCE["bytes_per_vector"]:.2f}x',
            ]
        )
    return table(['Route', 'bytes/image', 'bytes/vector', 'vs 8.35 KB reference'], rows)


def perf_points(doc: dict[str, Any]) -> str:
    rows = []
    for model, by_batch in doc['triton'].items():
        for batch, entry in sorted(by_batch.items(), key=lambda kv: int(kv[0])):
            best = entry['best']
            if not best:
                rows.append([f'`{model}`', batch, 'n/a', 'n/a', 'n/a', 'n/a'])
                continue
            first = entry['points'][0]
            rows.append(
                [
                    f'`{model}`',
                    batch,
                    f'{first["infer_per_sec"]:.0f}',
                    f'{first["p50_ms"]:.1f}',
                    f'{best["infer_per_sec"]:.0f}',
                    f'{best["concurrency"]:.0f}',
                ]
            )
    return table(
        [
            'Model',
            'batch',
            'infer/s at concurrency 1',
            'p50 ms at concurrency 1',
            'best infer/s',
            'at concurrency',
        ],
        rows,
    )


def endpoints(doc: dict[str, Any]) -> str:
    rows = []
    for name, by_threads in doc['endpoints'].items():
        if name == 'images':
            continue
        for threads, entry in by_threads.items():
            s = entry['summary']
            rows.append(
                [
                    f'`{name}`',
                    threads.removeprefix('threads_'),
                    fmt(s['images_per_s'], 1),
                    fmt(s['p50_ms'], 0),
                    fmt(s['p95_ms'], 0),
                ]
            )
    return table(['Endpoint', 'client threads', 'images/s', 'p50 ms', 'p95 ms'], rows)


SECTIONS = {
    'env': environment,
    'ingest': ingest_headline,
    'stages': ingest_stages,
    'models': triton_models,
    'gpu': gpu_rows,
    'storage': storage,
    'perf': perf_points,
    'endpoints': endpoints,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result', type=Path)
    p.add_argument('--section', choices=sorted(SECTIONS))
    args = p.parse_args(argv)
    doc = json.loads(args.result.read_text(encoding='utf-8'))
    for name, fn in SECTIONS.items():
        if args.section in (None, name):
            print(f'### {name}\n\n{fn(doc)}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

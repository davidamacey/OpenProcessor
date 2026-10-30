"""The per-stage table of one ingest profile row, from its repetition reports.

Pure: reads the JSON files ``run_baseline.py --reps`` wrote and renders markdown. Every number is
the median over the repetitions (range for images/s). Stage seconds are sums over concurrent
work: they are compared across rows and shares, never added up to wall time. The output holds
aggregates only, so it is safe to paste for the private set (numbers only).
"""

from __future__ import annotations

import json
import re
import statistics
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

API_STAGE = re.compile(r'op_pipeline_stage_seconds\{stage=(\w+)\}')
TRITON_SERIES = re.compile(
    r'(nv_inference_(?:queue_duration_us|compute_infer_duration_us|count))(?:_total)?\{[^}]*?model=([^,}]+)'
)


def _rep_number(path: Path) -> int:
    match = re.search(r'rep(\d+)', path.name)
    if match is None:
        raise ValueError(f'{path.name} is not a repetition report')
    return int(match.group(1))


def load_reps(directory: Path) -> list[dict[str, Any]]:
    """Every ``*-rep<N>.json`` of a row directory, in repetition order."""
    files = sorted(directory.glob('*-rep*.json'), key=_rep_number)
    if not files:
        raise FileNotFoundError(f'no repetition reports in {directory}')
    return [json.loads(f.read_text(encoding='utf-8')) for f in files]


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def api_stage_rows(reports: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """``(stage, seconds per image, share of the stage seconds)``, largest first."""
    per_rep: list[dict[str, float]] = []
    for r in reports:
        hist = r['metrics'].get('api', {}).get('histograms', {})
        images = r['ingest']['images'] or 1
        per_rep.append(
            {
                m.group(1): v['sum'] / images
                for k, v in hist.items()
                if (m := API_STAGE.fullmatch(k))
            }
        )
    stages = sorted({s for rep in per_rep for s in rep})
    med = {s: _median([rep.get(s, 0.0) for rep in per_rep]) for s in stages}
    total = sum(med.values()) or 1.0
    return sorted(((s, med[s], med[s] / total) for s in stages), key=lambda row: -row[1])


def triton_model_rows(reports: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """``(model, queue ms per request, compute ms per request)`` for every model that ran."""
    per_rep: list[dict[str, dict[str, float]]] = []
    for r in reports:
        models: dict[str, dict[str, float]] = {}
        for key, value in r['metrics'].get('triton', {}).get('counters', {}).items():
            if m := TRITON_SERIES.match(key):
                series = models.setdefault(m.group(2), {})
                series[m.group(1)] = series.get(m.group(1), 0.0) + value
        per_rep.append(models)
    rows = []
    for model in sorted({m for rep in per_rep for m in rep}):
        queue, compute = [], []
        for rep in per_rep:
            s = rep.get(model, {})
            n = s.get('nv_inference_count', 0.0)
            if n:
                queue.append(s.get('nv_inference_queue_duration_us', 0.0) / n / 1000)
                compute.append(s.get('nv_inference_compute_infer_duration_us', 0.0) / n / 1000)
        if queue:
            rows.append((model, _median(queue), _median(compute)))
    return rows


def prep_stage_rows(reports: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """``(stage, mean ms per image, share)`` of the gpu_image_prep stages."""
    stages = sorted({s for r in reports for s in r.get('gpu_prep_stages', {})})
    return [
        (
            s,
            _median(
                [
                    r['gpu_prep_stages'][s]['mean_s'] * 1000
                    for r in reports
                    if s in r.get('gpu_prep_stages', {})
                ]
            ),
            _median(
                [
                    r['gpu_prep_stages'][s]['share']
                    for r in reports
                    if s in r.get('gpu_prep_stages', {})
                ]
            ),
        )
        for s in stages
    ]


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return [*lines, '']


def render_row(name: str, reports: list[dict[str, Any]], gpu_index: str = '0') -> str:
    """Markdown for one row: headline, API stages, Triton models, gpu_image_prep stages."""
    rates = [r['ingest']['images_per_s'] for r in reports]
    images = reports[0]['ingest']['images']
    gpu = [r['gpu'][gpu_index] for r in reports]  # the stack's GPU; the sampler logs all of them
    noisy = [i for i, r in enumerate(reports, 1) if r.get('host', {}).get('noisy')]
    host = [r.get('host', {}) for r in reports]
    cpu = [
        h['api_cpu_s'] / r['ingest']['images']
        for h, r in zip(host, reports, strict=True)
        if h.get('api_cpu_s')
    ]
    cores = [h['triton_cpu_pct_mean'] / 100 for h in host if 'triton_cpu_pct_mean' in h]
    crops = _median(
        [r['ingest']['items'] / r['ingest']['images'] for r in reports if r['ingest']['images']]
    )
    lines = [
        f'### {name}',
        '',
        *_table(
            ('measure', 'value'),
            [
                ('repetitions', str(len(reports))),
                ('images per repetition', f'{images}'),
                (
                    'images/s median (min to max)',
                    f'{_median(rates):.2f} ({min(rates):.2f} to {max(rates):.2f})',
                ),
                ('crops per image', f'{crops:.2f}'),
                (
                    'GPU util mean / p50 %',
                    f'{_median([g["util_mean_pct"] for g in gpu]):.0f} / {_median([g.get("util_p50_pct", 0.0) for g in gpu]):.0f}',
                ),
                (
                    'Triton cores (docker stats)',
                    f'{_median(cores):.2f}' if cores else 'not recorded',
                ),
                ('API CPU-s per image', f'{_median(cpu):.4f}' if cpu else 'not recorded'),
                (
                    '1-minute load at start (max)',
                    f'{max(r["host"]["loadavg_start"][0] for r in reports):.1f}',
                ),
                ('noisy repetitions', ', '.join(map(str, noisy)) or 'none'),
            ],
        ),
        '#### API stages (op_pipeline_stage_seconds, sums over concurrent work)',
        '',
        *_table(
            ('stage', 'ms / image', 'share'),
            [(s, f'{v * 1000:.2f}', f'{sh * 100:.1f}%') for s, v, sh in api_stage_rows(reports)],
        ),
        '#### Triton models',
        '',
        *_table(
            ('model', 'queue ms / request', 'compute ms / request'),
            [(m, f'{q:.2f}', f'{c:.2f}') for m, q, c in triton_model_rows(reports)],
        ),
    ]
    prep = prep_stage_rows(reports)
    if prep:
        lines += [
            '#### gpu_image_prep stages (host wall, sums over concurrent requests)',
            '',
            *_table(
                ('stage', 'ms / image', 'share'),
                [(s, f'{ms:.2f}', f'{sh * 100:.1f}%') for s, ms, sh in prep],
            ),
        ]
    return '\n'.join(lines)

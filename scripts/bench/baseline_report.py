"""Pure logic for the baseline harness: metric deltas, rates, GPU summary, report, compare.

No network and no filesystem here, so every number a baseline report prints
is unit-tested. A report never holds image content or image paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from prometheus_client.parser import text_string_to_metric_families


SCHEMA = 1


def _series_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    return name + '{' + ','.join(f'{k}={v}' for k, v in sorted(labels.items())) + '}'


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus exposition text -> ``{series: value}`` (histogram buckets dropped)."""
    series: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.endswith('_bucket') or sample.name.endswith('_created'):
                continue
            series[_series_key(sample.name, dict(sample.labels))] = float(sample.value)
    return series


def metrics_delta(
    before: dict[str, float], after: dict[str, float], prefixes: tuple[str, ...]
) -> dict[str, Any]:
    """Per-series change over a run, for series whose name starts with a prefix.

    ``*_sum``/``*_count`` pairs become a histogram entry (sum, count, mean);
    everything else is a counter delta. A series first seen after the run
    started counts from zero.
    """
    deltas = {
        key: value - before.get(key, 0.0)
        for key, value in after.items()
        if key.startswith(prefixes)
    }
    histograms: dict[str, dict[str, float]] = {}
    counters: dict[str, float] = {}
    for key, value in sorted(deltas.items()):
        name, brace, labels = key.partition('{')
        if name.endswith('_sum') and name[: -len('_sum')] + '_count' + brace + labels in deltas:
            base = name[: -len('_sum')] + brace + labels
            count = deltas[name[: -len('_sum')] + '_count' + brace + labels]
            histograms[base] = {
                'sum': value,
                'count': count,
                'mean': value / count if count else 0.0,
            }
        elif (
            not name.endswith('_count')
            or name[: -len('_count')] + '_sum' + brace + labels not in deltas
        ):
            counters[key] = value
    return {'histograms': histograms, 'counters': counters}


def parse_nvidia_smi(text: str) -> dict[str, tuple[float, float]]:
    """``index, util %, memory MiB`` CSV lines -> ``{index: (util, mem_mb)}``."""
    out: dict[str, tuple[float, float]] = {}
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3:
            out[parts[0]] = (float(parts[1]), float(parts[2]))
    return out


def summarize_gpu(samples: list[dict[str, tuple[float, float]]]) -> dict[str, dict[str, float]]:
    """Mean utilization and peak memory per GPU over the sampled run."""
    per_gpu: dict[str, list[tuple[float, float]]] = {}
    for sample in samples:
        for gpu, reading in sample.items():
            per_gpu.setdefault(gpu, []).append(reading)
    return {
        gpu: {
            'util_mean_pct': sum(u for u, _ in readings) / len(readings),
            'mem_peak_mb': max(m for _, m in readings),
            'samples': len(readings),
        }
        for gpu, readings in per_gpu.items()
    }


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass(frozen=True)
class BatchOutcome:
    images: int
    ok: int
    duplicates: int
    failed: int
    items: int
    embedded: int
    seconds: float


def ingest_summary(
    batches: list[BatchOutcome], *, wall_s: float, input_bytes: int
) -> dict[str, float]:
    """Images/s, items/s and batch latency over the measured (post-warmup) batches."""
    images = sum(b.images for b in batches)
    items = sum(b.items for b in batches)
    seconds = [b.seconds for b in batches]
    return {
        'images': images,
        'ok': sum(b.ok for b in batches),
        'duplicates': sum(b.duplicates for b in batches),
        'failed': sum(b.failed for b in batches),
        'items': items,
        'embedded': sum(b.embedded for b in batches),
        'wall_s': wall_s,
        'images_per_s': images / wall_s if wall_s > 0 else 0.0,
        'items_per_s': items / wall_s if wall_s > 0 else 0.0,
        'input_bytes': input_bytes,
        'input_mb_per_s': input_bytes / 1e6 / wall_s if wall_s > 0 else 0.0,
        'batch_s_p50': percentile(seconds, 50),
        'batch_s_p99': percentile(seconds, 99),
    }


def storage_summary(*, images: int, store_bytes: int, crop_cache_bytes: int) -> dict[str, float]:
    return {
        'store_bytes': store_bytes,
        'crop_cache_bytes': crop_cache_bytes,
        'store_bytes_per_image': store_bytes / images if images else 0.0,
        'crop_cache_bytes_per_image': crop_cache_bytes / images if images else 0.0,
    }


def _flatten(node: Any, prefix: str = '') -> dict[str, float]:
    if isinstance(node, dict):
        out: dict[str, float] = {}
        for key, value in node.items():
            out.update(_flatten(value, f'{prefix}.{key}' if prefix else str(key)))
        return out
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return {prefix: float(node)}
    return {}


def _fmt(value: float) -> str:
    return f'{round(value, 6):g}'


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Markdown delta table over every numeric field present in both reports."""
    flat_before = _flatten({k: v for k, v in before.items() if k != 'schema'})
    flat_after = _flatten({k: v for k, v in after.items() if k != 'schema'})
    rows = ['| metric | before | after | delta | change |', '|---|---:|---:|---:|---:|']
    for key in sorted(flat_before.keys() & flat_after.keys()):
        b, a = flat_before[key], flat_after[key]
        delta = a - b
        delta_text = '0' if delta == 0 else f'{round(delta, 6):+g}'
        change = f'{delta / b * 100:+.1f}%' if b else 'n/a'
        rows.append(f'| {key} | {_fmt(b)} | {_fmt(a)} | {delta_text} | {change} |')
    return '\n'.join(rows)


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return [*lines, '']


def to_markdown(report: dict[str, Any]) -> str:
    """Human-readable tables for one report."""
    ingest = report.get('ingest', {})
    lines = ['## Baseline report', '']
    manifest = report.get('manifest', {})
    lines += [
        f'Manifest `{manifest.get("name")}` sha256 `{manifest.get("sha256")}`, '
        f'{manifest.get("count")} images, {manifest.get("bytes")} bytes. '
        f'Config: {report.get("config")}',
        '',
    ]
    lines += _table(
        ('measure', 'value'),
        [
            ('images/s', _fmt(ingest.get('images_per_s', 0.0))),
            ('items/s', _fmt(ingest.get('items_per_s', 0.0))),
            ('input MB/s', _fmt(ingest.get('input_mb_per_s', 0.0))),
            ('batch p50 s', _fmt(ingest.get('batch_s_p50', 0.0))),
            ('batch p99 s', _fmt(ingest.get('batch_s_p99', 0.0))),
            ('failed', _fmt(ingest.get('failed', 0.0))),
        ],
    )
    stages = report.get('stages', {})
    if stages:
        lines += ['### Stage wall time', '']
        lines += _table(
            ('stage', 'wall s'), [(k, _fmt(float(v.get('wall_s', 0.0)))) for k, v in stages.items()]
        )
    storage = report.get('storage', {})
    lines += ['### Storage after settle', '']
    lines += _table(('measure', 'value'), [(k, _fmt(float(v))) for k, v in storage.items()])
    gpu = report.get('gpu', {})
    if gpu:
        lines += ['### GPU', '']
        lines += _table(
            ('gpu', 'mean util %', 'peak mem MB'),
            [(g, _fmt(v['util_mean_pct']), _fmt(v['mem_peak_mb'])) for g, v in sorted(gpu.items())],
        )
    for source, block in sorted(report.get('metrics', {}).items()):
        hist = block.get('histograms', {})
        if hist:
            lines += [f'### Per-stage timers ({source})', '']
            lines += _table(
                ('series', 'count', 'sum s', 'mean s'),
                [
                    (k, _fmt(v['count']), _fmt(v['sum']), _fmt(v['mean']))
                    for k, v in sorted(hist.items())
                ],
            )
        counters = block.get('counters', {})
        if counters:
            lines += [f'### Counters ({source})', '']
            lines += _table(
                ('series', 'delta'), [(k, _fmt(v)) for k, v in sorted(counters.items())]
            )
    return '\n'.join(lines)

"""Pure logic for the baseline suite: run statistics, Triton statistics deltas,
perf_analyzer CSV parsing, OpenSearch storage accounting, GPU attribution.

No network, no subprocess, no filesystem: every number the suite reports goes
through a function here that a unit test pins down.
"""

from __future__ import annotations

import csv
import io
import statistics
from typing import Any


SCHEMA = 2
DTYPE_BYTES = {
    'BOOL': 1,
    'UINT8': 1,
    'INT8': 1,
    'UINT16': 2,
    'INT16': 2,
    'FP16': 2,
    'BF16': 2,
    'UINT32': 4,
    'INT32': 4,
    'FP32': 4,
    'UINT64': 8,
    'INT64': 8,
    'FP64': 8,
}


def summarize(values: list[float]) -> dict[str, float | int]:
    """n, median, mean, min, max and spread (max - min as percent of the median)."""
    if not values:
        return {'n': 0}
    median = statistics.median(values)
    return {
        'n': len(values),
        'median': median,
        'mean': statistics.fmean(values),
        'min': min(values),
        'max': max(values),
        'spread_pct': (max(values) - min(values)) / median * 100.0 if median else 0.0,
    }


def summarize_runs(runs: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    """Per-key :func:`summarize` over a list of flat ``{metric: number}`` dicts."""
    keys = sorted({k for run in runs for k in run})
    return {k: summarize([run[k] for run in runs if k in run]) for k in keys}


def parse_triton_stats(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``GET /v2/models/stats`` -> ``{model: counters}`` with ints, not strings.

    ``queue_ns`` is the total time requests waited. ``batches`` holds the
    per-execution figures keyed by batch size: ``count`` executions and the
    ``input_ns``/``infer_ns``/``output_ns`` they took in total. (Triton's own
    per-request compute figures credit every request in a batch with the whole
    execution, so they overcount by the requests per batch; they are not used.)
    """
    out: dict[str, dict[str, Any]] = {}
    for entry in doc.get('model_stats', []):
        stats = entry.get('inference_stats', {})
        row: dict[str, Any] = {
            'inference_count': int(entry.get('inference_count', 0)),
            'execution_count': int(entry.get('execution_count', 0)),
            'request_count': int(stats.get('success', {}).get('count', 0)),
            'batches': {
                int(b['batch_size']): {
                    'count': int(b['compute_infer']['count']),
                    'input_ns': int(b['compute_input']['ns']),
                    'infer_ns': int(b['compute_infer']['ns']),
                    'output_ns': int(b['compute_output']['ns']),
                }
                for b in entry.get('batch_stats', [])
            },
        }
        row['queue_ns'] = int(stats.get('queue', {}).get('ns', 0))
        out[entry['name']] = row
    return out


def triton_stats_delta(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per-model change over a run; models that did no work are dropped.

    Per-request: ``queue_ms_per_request`` is the time a request waited for the
    batcher. Per-execution (``exec_*_ms``, from the batch statistics) is the
    time of one batch in each Triton phase. The TensorRT backend overlaps one
    batch's output wait with the next batch's enqueue, so input + infer + output
    over-counts the device time; use ``exec_infer_ms`` x ``executions``.
    """
    out: dict[str, dict[str, Any]] = {}
    for model, a in after.items():
        b = before.get(model, {})
        inferences = a['inference_count'] - b.get('inference_count', 0)
        executions = a['execution_count'] - b.get('execution_count', 0)
        requests = a['request_count'] - b.get('request_count', 0)
        if inferences <= 0:
            continue
        before_batches = b.get('batches', {})
        sizes = {
            size: {
                k: v[k] - before_batches.get(size, {}).get(k, 0)
                for k in ('count', 'input_ns', 'infer_ns', 'output_ns')
            }
            for size, v in sorted(a['batches'].items())
        }
        sizes = {size: v for size, v in sizes.items() if v['count'] > 0}
        phase_ns = {
            ph: sum(v[f'{ph}_ns'] for v in sizes.values()) for ph in ('input', 'infer', 'output')
        }
        row: dict[str, Any] = {
            'inferences': inferences,
            'executions': executions,
            'requests': requests,
            'mean_batch': inferences / executions if executions else 0.0,
            'queue_ms_per_request': (a['queue_ns'] - b.get('queue_ns', 0)) / 1e6 / requests
            if requests
            else 0.0,
            'batch_histogram': {str(size): v['count'] for size, v in sizes.items()},
        }
        for ph, ns in phase_ns.items():
            row[f'exec_{ph}_ms'] = ns / 1e6 / executions if executions else 0.0
        out[model] = row
    return out


def tensor_bytes(dims: list[int], datatype: str, batch: int = 1) -> int:
    """Bytes of one tensor of ``batch`` x ``dims`` (variable dims, -1, count as 1)."""
    elems = batch
    for d in dims:
        elems *= max(d, 1)
    return elems * DTYPE_BYTES.get(datatype.removeprefix('TYPE_'), 4)


def triton_wire_bytes(
    delta: dict[str, dict[str, Any]], configs: dict[str, dict[str, Any]]
) -> dict[str, dict[str, int]]:
    """Estimated request/response payload bytes per model from the inference counts.

    ``configs`` is ``GET /v2/models/<m>/config`` per model. Variable-size inputs
    are counted at one element per variable dimension, so treat those as a floor.
    """
    out: dict[str, dict[str, int]] = {}
    for model, row in delta.items():
        cfg = configs.get(model)
        if not cfg:
            continue
        per_in = sum(tensor_bytes(t['dims'], t['data_type']) for t in cfg.get('input', []))
        per_out = sum(tensor_bytes(t['dims'], t['data_type']) for t in cfg.get('output', []))
        out[model] = {
            'request_bytes': per_in * row['inferences'],
            'response_bytes': per_out * row['inferences'],
        }
    return out


def parse_perf_csv(text: str) -> list[dict[str, float]]:
    """perf_analyzer ``-f`` CSV -> one dict per concurrency level (latencies in ms)."""
    rows = []
    for rec in csv.DictReader(io.StringIO(text)):
        row = {
            'concurrency': float(rec['Concurrency']),
            'infer_per_sec': float(rec['Inferences/Second']),
        }
        for label, key in (('p50', 'p50 latency'), ('p95', 'p95 latency'), ('p99', 'p99 latency')):
            if rec.get(key):
                row[f'{label}_ms'] = float(rec[key]) / 1000.0
        for label, key in (
            ('queue', 'Server Queue'),
            ('compute_input', 'Server Compute Input'),
            ('compute_infer', 'Server Compute Infer'),
            ('compute_output', 'Server Compute Output'),
        ):
            if rec.get(key):
                row[f'{label}_ms'] = float(rec[key]) / 1000.0
        rows.append(row)
    return rows


def best_point(rows: list[dict[str, float]]) -> dict[str, float]:
    """The concurrency level with the highest throughput."""
    return max(rows, key=lambda r: r['infer_per_sec'])


def storage_summary(cat_rows: list[dict[str, Any]], *, images: int, vectors: int) -> dict[str, Any]:
    """Bytes per image and per vector from ``_cat/indices?format=json&bytes=b`` rows.

    Rows with no documents are listed but add nothing. ``bytes`` is the primary
    store size (no replicas on a single node).
    """
    indexes = {
        r['index']: {
            'docs': int(r['docs.count']),
            'deleted': int(r['docs.deleted']),
            'bytes': int(r['pri.store.size']),
        }
        for r in cat_rows
    }
    total = sum(v['bytes'] for v in indexes.values())
    return {
        'indexes': indexes,
        'total_bytes': total,
        'images': images,
        'vectors': vectors,
        'bytes_per_image': total / images if images else 0.0,
        'bytes_per_vector': total / vectors if vectors else 0.0,
    }


def compare_storage(measured: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    """Ratio of each measured number to its reference (keys in both)."""
    return {k: measured[k] / reference[k] for k in reference if k in measured and reference[k]}


def foreign_gpu_summary(samples: list[dict[str, float]], own_pids: set[int]) -> dict[str, float]:
    """Utilization attributed to processes outside the stack, from pmon samples.

    Each sample maps pid -> sm percent (``-`` already dropped). Returns mean and
    peak of the foreign sum and of the own sum, and the share of samples in which
    any foreign process showed more than 0 percent.
    """
    if not samples:
        return {}
    foreign = [sum(v for pid, v in s.items() if pid not in own_pids) for s in samples]
    own = [sum(v for pid, v in s.items() if pid in own_pids) for s in samples]
    return {
        'samples': len(samples),
        'foreign_sm_mean_pct': statistics.fmean(foreign),
        'foreign_sm_peak_pct': max(foreign),
        'foreign_active_share': sum(1 for f in foreign if f > 0) / len(foreign),
        'own_sm_mean_pct': statistics.fmean(own),
        'own_sm_peak_pct': max(own),
    }


def parse_pmon(text: str, gpu: str = '0') -> dict[int, float]:
    """``nvidia-smi pmon -s u -c 1`` -> ``{pid: sm %}`` for compute processes on ``gpu``."""
    out: dict[int, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith('#') or len(parts) < 5:
            continue
        if parts[0] != gpu or parts[2] != 'C' or parts[3] == '-':
            continue
        out[int(parts[1])] = float(parts[3])
    return out

"""Per-model bake-off report (schema v2): assembly, ``summary.csv`` row, MLflow.

The report is one comparison row (see ``compare.py``) without ``rank``;
``common`` is left ``None`` here because it depends on the other models in
the comparison and is filled in by ``compare.build_comparison``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import argparse

    from .class_map import ClassMapping


SCHEMA_VERSION = 2


def weights_size_mb(weights: str | None) -> float | None:
    """On-disk size of a weights file/dir in MB (recursive for .mlpackage dirs)."""
    if not weights:
        return None
    path = Path(weights)
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 3)
    if path.is_dir():  # CoreML .mlpackage is a directory
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return round(total / (1024 * 1024), 3)
    return None


def build_report(
    args: argparse.Namespace,
    *,
    runtime: str,
    blocks: dict[str, Any],
    mapping: ClassMapping,
    latency_ms: dict[str, float],
    size_mb: float | None,
    frames: tuple[int, int, int],
) -> dict[str, Any]:
    """Assemble the per-model report v2 from the scoring blocks and run args."""
    test_frames, positive_frames, background_frames = frames
    return {
        'schema_version': SCHEMA_VERSION,
        'model': args.model_key,
        'display_name': args.display_name,
        'source': args.source,
        'run_id': args.run_id,
        'runtime': runtime,
        'imgsz': args.imgsz,
        'training_data': args.training_data,
        'overall': blocks['overall'],
        'common': None,
        'per_class': blocks['per_class'],
        'coverage': blocks['coverage'],
        'class_mapping': {'method': mapping.method, 'warnings': list(mapping.warnings)},
        'train_test_overlap': args.train_test_overlap,
        'latency_ms': latency_ms,
        'fps': (1000.0 / latency_ms['mean']) if latency_ms['mean'] > 0 else 0.0,
        'size_mb': size_mb,
        'per_stratum': blocks['per_stratum'],
        'test_frames': test_frames,
        'positive_frames': positive_frames,
        'background_frames': background_frames,
    }


def write_report(out_dir: Path, stem: str, report: dict[str, Any]) -> Path:
    """Write ``<out_dir>/<stem>.json`` and append the shared ``summary.csv`` row."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{stem}.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_summary_row(out_dir / 'summary.csv', report)
    return path


def log_mlflow(report: dict[str, Any], args: argparse.Namespace) -> None:
    """Optionally log this run to MLflow so its UI charts the comparison.

    Flag-gated and import-guarded: if mlflow isn't installed we just skip,
    keeping it off the harness's hard dependencies.
    """
    try:
        import mlflow
    except ImportError:
        print('mlflow not installed; skipping (.venv/bin/pip install mlflow)')
        return
    try:
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=report['display_name']):
            mlflow.log_params(
                {
                    'model': report['model'],
                    'runtime': report['runtime'],
                    'imgsz': report['imgsz'],
                    'training_data': args.training_data or 'unknown',
                    'class_mapping': report['class_mapping']['method'],
                }
            )
            metrics = {k: v for k, v in _summary_metrics(report).items() if v is not None}
            for row in report['per_class']:
                if row['covered']:
                    metrics[f'ap50_95/{_metric_key(row["name"])}'] = float(row['ap50_95'])
            mlflow.log_metrics(metrics)
            for stratum, vals in report.get('per_stratum', {}).items():
                if vals['map_50'] is not None:
                    mlflow.log_metric(f'stratum_map50/{_metric_key(stratum)}', vals['map_50'])
            # Artifact upload is best-effort: it needs the server's
            # --serve-artifacts proxy and can fail on artifact-root perms
            # without invalidating the (already-committed) metrics.
            try:
                mlflow.log_dict(report, f'{_metric_key(report["model"])}.json')
            except Exception as art_exc:
                print(f'mlflow artifact upload skipped: {art_exc}')
        print(f'logged to mlflow: {args.mlflow_uri} (experiment={args.mlflow_experiment})')
    except Exception as exc:  # server unreachable / transient — don't fail the run
        print(f'mlflow logging skipped: {exc}')


def _metric_key(text: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_./' else '_' for ch in str(text))


def _summary_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    o = report['overall']
    return {
        **{k: o[k] for k in _SUMMARY_FIELDS[5:13]},
        'ap_medium': o['ap_medium'],
        'ap_large': o['ap_large'],
        'latency_mean_ms': report['latency_ms']['mean'],
        'throughput_fps': report['fps'],
    }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


_SUMMARY_FIELDS = [
    'model',
    'display_name',
    'runtime',
    'imgsz',
    'n_classes',
    'map_50_95',
    'map_50',
    'map_75',
    'ap_small',
    'precision',
    'recall',
    'f1',
    'mean_iou',
    'latency_mean_ms',
    'throughput_fps',
]


def write_summary_row(path: Path, report: dict[str, Any]) -> None:
    """Append one row to a shared summary.csv so models accumulate."""
    m = _summary_metrics(report)
    row = {
        'model': report['model'],
        'display_name': report['display_name'],
        'runtime': report['runtime'],
        'imgsz': report['imgsz'],
        'n_classes': report['overall']['n_classes'],
        **{k: _round(m[k], 4) for k in _SUMMARY_FIELDS[5:13]},
        'latency_mean_ms': _round(m['latency_mean_ms'], 2),
        'throughput_fps': _round(m['throughput_fps'], 2),
    }
    exists = path.is_file()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

"""Per-model bake-off report: assembly, ``summary.csv`` row, MLflow logging.

Moved out of ``run.py`` so the CLI module stays about the scoring flow.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import argparse


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
        with mlflow.start_run(run_name=report['model']):
            mlflow.log_params(
                {
                    'model': report['model'],
                    'runtime': report['runtime'],
                    'imgsz': report['imgsz'],
                    'training_data': args.training_data or 'unknown',
                }
            )
            mlflow.log_metrics(_summary_metrics(report))
            for stratum, vals in report.get('per_stratum', {}).items():
                key = ''.join(ch if ch.isalnum() or ch in '-_./' else '_' for ch in stratum)
                mlflow.log_metric(f'stratum_map50/{key}', float(vals['map_50']))
            # Artifact upload is best-effort: it needs the server's
            # --serve-artifacts proxy and can fail on artifact-root perms
            # without invalidating the (already-committed) metrics.
            try:
                safe = report['model'].replace('/', '_').replace(' ', '_')
                mlflow.log_dict(report, f'{safe}.json')
            except Exception as art_exc:
                print(f'mlflow artifact upload skipped: {art_exc}')
        print(f'logged to mlflow: {args.mlflow_uri} (experiment={args.mlflow_experiment})')
    except Exception as exc:  # server unreachable / transient — don't fail the run
        print(f'mlflow logging skipped: {exc}')


def _summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    c, op, lat = report['coco'], report['operating_point'], report['latency_ms']
    return {
        'map_50_95': c['map_50_95'],
        'map_50': c['map_50'],
        'map_75': c['map_75'],
        'ap_small': c['ap_small'],
        'ap_medium': c['ap_medium'],
        'ap_large': c['ap_large'],
        'precision': op['precision'],
        'recall': op['recall'],
        'f1': op['f1'],
        'mean_iou': op['mean_iou'],
        'latency_mean_ms': lat['mean'],
        'throughput_fps': report['throughput_fps'],
    }


_SUMMARY_FIELDS = [
    'model',
    'runtime',
    'imgsz',
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
        'runtime': report['runtime'],
        'imgsz': report['imgsz'],
        **{k: round(m[k], 4) for k in _SUMMARY_FIELDS[3:11]},
        'latency_mean_ms': round(m['latency_mean_ms'], 2),
        'throughput_fps': round(m['throughput_fps'], 2),
    }
    exists = path.is_file()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

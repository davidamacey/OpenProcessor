"""Aggregate per-model bake-off JSON into comparison tables.

Each ``run`` writes ``<model>.json`` + a row in ``summary.csv``. This
merges every per-model JSON in a results dir into (a) a ranked markdown
table for quick reading, (b) a combined ``comparison.json`` the UI/API
serve, and (c) LaTeX table rows to paste into the paper's results table.

CLI:
    python -m scripts.curation.bakeoff.compare --results-dir /data/bakeoff
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_reports(results_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for f in sorted(results_dir.glob('*.json')):
        if f.name == 'comparison.json':
            continue
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue
        if 'coco' in data and 'operating_point' in data:
            reports.append(data)
    return reports


def _row(r: dict[str, Any]) -> dict[str, Any]:
    c, op = r['coco'], r['operating_point']
    return {
        'model': r['model'],
        'runtime': r.get('runtime', ''),
        'training_data': r.get('training_data', ''),
        'imgsz': r.get('imgsz', ''),
        'map_50': c['map_50'],
        'map_50_95': c['map_50_95'],
        'ap_small': c.get('ap_small', -1.0),
        'mean_iou': op.get('mean_iou', 0.0),
        'precision': op['precision'],
        'recall': op['recall'],
        'f1': op['f1'],
        'latency_ms': r.get('latency_ms', {}).get('mean', 0.0),
        'size_mb': r.get('size_mb'),
        'fps': r.get('throughput_fps', 0.0),
    }


def build_comparison(results_dir: Path) -> dict[str, Any]:
    """Rank models by mAP@.5:.95 and return a UI/API-ready structure."""
    rows = [_row(r) for r in _load_reports(results_dir)]
    rows.sort(key=lambda x: x['map_50_95'], reverse=True)
    return {'models': rows, 'n_models': len(rows)}


def to_markdown(comparison: dict[str, Any]) -> str:
    head = (
        '| Model | Runtime | Training data | mAP@.5 | mAP@.5:.95 | AP_s | '
        'meanIoU | P | R | F1 | ms |\n'
        '|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|\n'
    )
    lines = [
        f'| {m["model"]} | {m["runtime"]} | {m["training_data"] or "—"} '
        f'| {m["map_50"]:.3f} | {m["map_50_95"]:.3f} | {m["ap_small"]:.3f} '
        f'| {m["mean_iou"]:.3f} | {m["precision"]:.3f} | {m["recall"]:.3f} '
        f'| {m["f1"]:.3f} | {m["latency_ms"]:.1f} |'
        for m in comparison['models']
    ]
    return head + '\n'.join(lines) + '\n'


def to_latex_rows(comparison: dict[str, Any]) -> str:
    """LaTeX rows for the paper's headline table (Table 2 column order)."""
    out: list[str] = []
    for m in comparison['models']:
        name = m['model'].replace('_', r'\_')
        out.append(
            f'{name} & {m["map_50"]:.3f} & {m["map_50_95"]:.3f} & '
            f'{m["ap_small"]:.3f} & {m["mean_iou"]:.3f} & {m["precision"]:.3f} & '
            f'{m["recall"]:.3f} & {m["f1"]:.3f} & {m["latency_ms"]:.1f} \\\\'
        )
    return '\n'.join(out) + '\n'


def main() -> int:
    p = argparse.ArgumentParser(description='Aggregate bake-off results.')
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--latex', action='store_true', help='Also print LaTeX table rows')
    args = p.parse_args()

    comparison = build_comparison(args.results_dir)
    if not comparison['models']:
        raise SystemExit(f'no per-model JSON reports found in {args.results_dir}')

    (args.results_dir / 'comparison.json').write_text(
        json.dumps(comparison, indent=2), encoding='utf-8'
    )
    md = to_markdown(comparison)
    (args.results_dir / 'comparison.md').write_text(md, encoding='utf-8')
    print(md)
    if args.latex:
        print('% --- paste into Table 2 ---')
        print(to_latex_rows(comparison))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

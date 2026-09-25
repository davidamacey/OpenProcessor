"""Aggregate per-model bake-off reports (schema v2) into comparisons + a matrix.

Each ``run`` writes ``<model stem>.json`` into the dataset's results dir.
:func:`build_comparison` merges them into one ranked, class-aware
comparison for that dataset; :func:`build_matrix` puts several datasets'
comparisons side by side (model x dataset) with the winners per metric.

Comparison semantics:

* ``common_classes`` = eval classes covered by every model that covers at
  least one class. Every row gets a ``common`` block (its metrics over those
  classes) next to its ``overall`` block (its own covered classes).
* Rank on ``common`` (``rank_scope = "common"``); if no class is common, rank
  on ``overall`` and say so in ``warnings`` -- such numbers are not
  comparable.
* Ties = equal after rounding to 4 decimals; competition ranking (1, 1, 3);
  tied rows ordered by ``display_name``; rows with a null rank metric get
  ``rank: null`` and sort last.

CLI:
    python -m scripts.curation.bakeoff.compare --results-dir /data/bakeoff/<job>/<dataset>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .metrics import subset_scope
from .profile import RANKABLE_METRICS


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 2
_RESULT_FILES = {'comparison.json', 'matrix.json', 'status.json'}

# Matrix cell metrics. Accuracy metrics come from the dataset's rank scope.
MATRIX_METRICS = (
    'map_50',
    'map_50_95',
    'precision',
    'recall',
    'f1',
    'latency_ms',
    'size_mb',
    'coverage',
)
# Where the winning (bolded) cell is the minimum, not the maximum.
_LOWER_IS_BETTER = {'latency_ms', 'size_mb'}

NO_COMMON_WARNING = (
    "no class is covered by every model; ranked on each model's own classes, not comparable"
)


def _load_reports(results_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for f in sorted(results_dir.glob('*.json')):
        if f.name in _RESULT_FILES:
            continue
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get('schema_version') == SCHEMA_VERSION:
            reports.append(data)
    return reports


def _covered(report: Mapping[str, Any]) -> set[int]:
    return {r['eval_class_id'] for r in report['per_class'] if r['covered']}


def _rank_key(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _assign_ranks(rows: list[dict[str, Any]], scope: str, rank_by: str) -> list[dict[str, Any]]:
    keyed = [(_rank_key(r[scope][rank_by]), r) for r in rows]
    ranked = sorted(
        (kr for kr in keyed if kr[0] is not None),
        key=lambda kr: (-kr[0], str(kr[1]['display_name'])),  # type: ignore[operator]
    )
    unranked = sorted(
        (kr for kr in keyed if kr[0] is None), key=lambda kr: str(kr[1]['display_name'])
    )
    out: list[dict[str, Any]] = []
    prev: float | None = None
    rank = 0
    for i, (value, row) in enumerate(ranked, start=1):
        if value != prev:
            rank, prev = i, value
        out.append({'rank': rank, **row})
    out.extend({'rank': None, **row} for _, row in unranked)
    return out


def build_comparison(
    results_dir: Path,
    *,
    rank_by: str = 'map_50_95',
    dataset_meta: Mapping[str, Any] | None = None,
    job_id: str | None = None,
    profile: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    failed: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rank every v2 report in ``results_dir`` (one dataset) by ``rank_by``.

    ``dataset_meta`` carries the dataset identity (``id``, ``frozen_test_sha``,
    ``test_label_sha``); frame/object counts are filled from the reports when
    absent. ``failed`` lists models whose scoring failed (``{model, error}``).
    """
    if rank_by not in RANKABLE_METRICS:
        raise ValueError(f'cannot rank by {rank_by!r}; one of {", ".join(RANKABLE_METRICS)}')
    reports = _load_reports(results_dir)

    eval_classes: dict[int, dict[str, Any]] = {}
    for r in reports:
        for row in r['per_class']:
            eval_classes.setdefault(
                row['eval_class_id'],
                {'eval_class_id': row['eval_class_id'], 'name': row['name'], 'n_gt': row['n_gt']},
            )
    covering = [_covered(r) for r in reports if _covered(r)]
    common = sorted(set.intersection(*covering)) if covering else []
    warnings: list[str] = []
    scope = 'common'
    if not common:
        scope = 'overall'
        if len(reports) > 1:
            warnings.append(NO_COMMON_WARNING)

    rows = [{**r, 'common': subset_scope(r['per_class'], common)} for r in reports]
    models = _assign_ranks(rows, scope, rank_by)

    meta = dict(dataset_meta or {})
    first = reports[0] if reports else {}
    meta.setdefault('n_images', first.get('test_frames'))
    meta.setdefault('n_objects', sum(c['n_gt'] for c in eval_classes.values()))
    meta.setdefault('n_background_images', first.get('background_frames'))
    return {
        'schema_version': SCHEMA_VERSION,
        'job_id': job_id,
        'profile': profile,
        'thresholds': dict(thresholds or {}),
        'dataset': meta,
        'eval_classes': [eval_classes[k] for k in sorted(eval_classes)],
        'common_classes': common,
        'rank_by': rank_by,
        'rank_scope': scope,
        'models': models,
        'failed': [dict(f) for f in failed],
        'warnings': warnings,
        'n_models': len(models),
    }


def _cell(row: Mapping[str, Any], scope: str) -> dict[str, Any]:
    block = row[scope]
    cov = row['coverage']
    n_eval = cov['n_eval_classes']
    return {
        **{k: block.get(k) for k in ('map_50', 'map_50_95', 'precision', 'recall', 'f1')},
        'latency_ms': (row.get('latency_ms') or {}).get('mean'),
        'size_mb': row.get('size_mb'),
        'coverage': (cov['n_covered'] / n_eval) if n_eval else None,
        'rank': row.get('rank'),
    }


def build_matrix(
    dataset_entries: Sequence[Mapping[str, Any]],
    per_dataset: Mapping[str, Mapping[str, Any]],
    *,
    job_id: str | None = None,
    rank_by: str | None = None,
) -> dict[str, Any]:
    """Model x dataset matrix with the winners per (dataset, metric).

    ``dataset_entries`` = the job's datasets (``id``, ``frozen_test_sha``,
    ``test_label_sha``) in order; ``per_dataset[id]`` = that dataset's
    comparison (absent when it was not scored). ``best[id][metric]`` is the
    LIST of every tied winning model key (max, or min for latency/size).
    """
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    models: dict[str, dict[str, Any]] = {}
    datasets: list[dict[str, Any]] = []
    best: dict[str, dict[str, list[str]]] = {}
    for entry in dataset_entries:
        ds_id = str(entry['id'])
        comp = per_dataset.get(ds_id)
        datasets.append(
            {
                'id': ds_id,
                'frozen_test_sha': entry.get('frozen_test_sha'),
                'test_label_sha': entry.get('test_label_sha'),
                'rank_scope': comp['rank_scope'] if comp else None,
                'n_common_classes': len(comp['common_classes']) if comp else 0,
            }
        )
        best[ds_id] = {}
        if not comp:
            continue
        for row in comp['models']:
            key = row['model']
            models.setdefault(
                key,
                {'model': key, 'display_name': row['display_name'], 'source': row['source']},
            )
            cells.setdefault(key, {})[ds_id] = _cell(row, comp['rank_scope'])
        for metric in MATRIX_METRICS:
            scored = [
                (m, _rank_key(c[ds_id][metric]))
                for m, c in cells.items()
                if ds_id in c and isinstance(c[ds_id].get(metric), (int, float))
            ]
            if not scored:
                continue
            values = [v for _, v in scored]
            target = min(values) if metric in _LOWER_IS_BETTER else max(values)  # type: ignore[type-var]
            best[ds_id][metric] = [m for m, v in scored if v == target]
    return {
        'schema_version': SCHEMA_VERSION,
        'job_id': job_id,
        'rank_by': rank_by,
        'datasets': datasets,
        'models': list(models.values()),
        'metrics': list(MATRIX_METRICS),
        'cells': cells,
        'best': best,
    }


def _fmt(value: Any, spec: str = '.3f') -> str:
    return '—' if value is None else format(value, spec)


def to_markdown(comparison: Mapping[str, Any]) -> str:
    """Ranked table; accuracy columns from the comparison's rank scope."""
    scope = comparison['rank_scope']
    head = (
        f'| Rank | Model | Runtime | Classes | mAP@.5 ({scope}) | mAP@.5:.95 ({scope}) '
        '| P | R | F1 | ms |\n'
        '|--:|---|---|--:|--:|--:|--:|--:|--:|--:|\n'
    )
    lines = []
    for m in comparison['models']:
        block, cov = m[scope], m['coverage']
        lines.append(
            f'| {_fmt(m["rank"], "d")} | {m["display_name"]} | {m["runtime"]} '
            f'| {cov["n_covered"]}/{cov["n_eval_classes"]} '
            f'| {_fmt(block["map_50"])} | {_fmt(block["map_50_95"])} '
            f'| {_fmt(block["precision"])} | {_fmt(block["recall"])} | {_fmt(block["f1"])} '
            f'| {_fmt((m.get("latency_ms") or {}).get("mean"), ".1f")} |'
        )
    return head + '\n'.join(lines) + '\n'


def to_latex_rows(comparison: Mapping[str, Any]) -> str:
    """LaTeX table rows (model, mAP@.5, mAP@.5:.95, P, R, F1, ms) from the rank scope."""
    scope = comparison['rank_scope']
    out: list[str] = []
    for m in comparison['models']:
        block = m[scope]
        name = str(m['display_name']).replace('_', r'\_')
        out.append(
            f'{name} & {_fmt(block["map_50"])} & {_fmt(block["map_50_95"])} & '
            f'{_fmt(block["precision"])} & {_fmt(block["recall"])} & {_fmt(block["f1"])} & '
            f'{_fmt((m.get("latency_ms") or {}).get("mean"), ".1f")} \\\\'
        )
    return '\n'.join(out) + '\n'


def main() -> int:
    p = argparse.ArgumentParser(description='Aggregate bake-off results for one dataset.')
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--latex', action='store_true', help='Also print LaTeX table rows')
    p.add_argument('--rank-by', default='map_50_95', help='Metric to rank models by')
    args = p.parse_args()

    comparison = build_comparison(args.results_dir, rank_by=args.rank_by)
    if not comparison['models']:
        raise SystemExit(f'no per-model v2 reports found in {args.results_dir}')

    (args.results_dir / 'comparison.json').write_text(
        json.dumps(comparison, indent=2), encoding='utf-8'
    )
    md = to_markdown(comparison)
    (args.results_dir / 'comparison.md').write_text(md, encoding='utf-8')
    print(md)
    if args.latex:
        print('% --- LaTeX table rows ---')
        print(to_latex_rows(comparison))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Run the whole face-grouping evaluation on the stored embeddings (CPU only).

    python -I run_eval.py <eval_faces_dir> <out_dir>

``<eval_faces_dir>`` holds ``lfw.npz``, ``calfw.npz``, ``cplfw.npz`` (``extract_embeddings.py``) and the
copied set directories with their pairs files. Writes ``results.json`` and ``results.md`` into ``out_dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.research.face_grouping import (
    datasets as ds,
    evaluate as ev,
    report,
    scores as sc,
    sweep as sw,
)


TODAY = {'method': 'single', 'params': {'sim': 0.7}}


def dataset_facts(eval_dir: Path, sets: dict[str, ds.FaceSet]) -> dict[str, Any]:
    facts = {}
    for name, s in sets.items():
        facts[name] = {
            'images': len(s.has),
            'identities': len(set(s.identity.tolist())),
            'images_zero_faces': int((~s.has).sum()),
            'images_multiple_faces': int((s.n_faces > 1).sum()),
            'npz_sha256': ds.sha256(eval_dir / f'{name}.npz'),
            'manifest_sha256': ds.sha256(eval_dir / name / 'manifest.json'),
            'pairs_sha256': ds.sha256(eval_dir / name / ds.PAIRS[name]),
        }
    return facts


def best_by_method(rows: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    out = {}
    for m in sorted({r['method'] for r in rows}):
        out[m] = sc.best_point([r for r in rows if r['method'] == m], floor)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('eval_dir', type=Path)
    ap.add_argument('out_dir', type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sets = {n: ds.load_set(args.eval_dir, n) for n in ds.SETS}
    pooled = ds.pool(sets)
    result: dict[str, Any] = {
        'datasets': dataset_facts(args.eval_dir, sets),
        'overlap': ds.overlap_report(sets),
    }

    tables = {n: ev.pair_table(sets[n], ev.set_pairs(args.eval_dir, n)) for n in ds.SETS}
    result['verification'] = {n: ev.verification_report(t) for n, t in tables.items()}

    bed_rows: dict[str, list[dict[str, Any]]] = {}
    for bed, idx in ds.beds(pooled).items():
        rows = sw.sweep(pooled.emb[idx], pooled.identity[idx])
        bed_rows[bed] = rows
        print(f'{bed}: {len(idx)} faces, {len(rows)} sweep points', file=sys.stderr)
    result['beds'] = {
        bed: {
            'faces': len(ds.beds(pooled)[bed]),
            'sweep': rows,
            'best_at_0.95': best_by_method(rows, 0.95),
            'best_at_0.99': best_by_method(rows, 0.99),
        }
        for bed, rows in bed_rows.items()
    }

    pooled_rows = bed_rows['bed2_pooled']
    candidates = {m: p for m, p in best_by_method(pooled_rows, 0.95).items() if p and m != 'finch'}
    best_method = max(candidates, key=lambda m: candidates[m]['bcubed_recall'])
    spec = {'method': best_method, 'params': candidates[best_method]['params']}
    result['best_method_bed2_p95'] = spec

    partition = sw.run_spec(spec, pooled.emb)
    result['quality'] = ev.quality_report(pooled, tables, partition)
    signal = min(result['quality'], key=lambda k: result['quality'][k]['pair_error_reject_area'])
    result['quality_best_signal'] = signal
    result['gated'] = ev.gated_report(pooled, spec, signal)
    result['incremental'] = ev.incremental_report(pooled, spec)
    result['today_bed2'] = next(
        r for r in pooled_rows if r['method'] == 'single' and r['params'] == TODAY['params']
    )

    (args.out_dir / 'results.json').write_text(json.dumps(result, indent=1, default=float))
    (args.out_dir / 'results.md').write_text(report.markdown(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

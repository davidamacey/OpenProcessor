#!/usr/bin/env python3
"""Generate LaTeX numbers for the LPR bake-off paper from harness output.

Reads a completed bake-off matrix run (``matrix.json`` -- the aggregated
``cells[model][dataset][metric]`` table) and emits:

* ``numbers.tex`` -- ``\\newcommand`` macros for the inline metrics quoted in the
  abstract / discussion / conclusion, so the prose can never drift from the data.
* a printed report of the regenerated headline + cross-dataset table rows (with
  per-column bolding), ready to verify or paste into the paper.

Honors the rule in ``docs/paper/lpr_bakeoff.tex`` (never hand-type a metric; pull
it from the harness output). Run after every bake-off rerun::

    .venv/bin/python scripts/curation/bakeoff/paper_numbers.py \\
        --bakeoff-root /data/curation_train_data/bakeoff \\
        --out docs/paper/numbers.tex

Stdlib only -- safe to run anywhere.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('paper_numbers')

DEFAULT_BAKEOFF_ROOT = Path('/data/curation_train_data/bakeoff')

# Public benchmark dataset dir-names; anything else in a matrix run is "curated".
PUBLIC_DATASETS = ('andrewmvd_car_plate', 'openalpr_us', 'roboflow_alpr')
ACC_FIELDS = ('map_50', 'map_50_95', 'ap_small', 'mean_iou', 'precision', 'recall', 'f1')

# Quantization variants for tab:quant, in efficiency order. Each is scored as its
# own model row (produced by export/quantize.py + the Mac CoreML workflow); rows
# absent from a given matrix run are simply skipped.
QUANT_VARIANTS: tuple[tuple[str, str], ...] = (
    ('ours_fp32_onnx', 'FP32 ONNX (baseline)'),
    ('ours_fp16_onnx', 'FP16 ONNX (NVIDIA)'),
    ('ours_int8_onnx', 'INT8 QDQ ONNX'),
    ('ours_fp16_coreml', 'FP16 CoreML (Apple)'),
    ('ours_int8_coreml', 'INT8 CoreML (Apple)'),
)

# The six headline contenders, in paper order, each in the regime the paper shows
# (LPDNet in its design/crop regime; the rest full-frame).
HEADLINE_ROWS: tuple[tuple[str, str, str], ...] = (
    ('ours', '[full]', 'Ours (YOLO26)'),
    ('lpr_nanov11_640', '[full]', 'lpr\\_nanov11\\_640'),
    ('open-image-models', '[full]', 'open-image-models'),
    ('morsetechlab', '[full]', 'morsetechlab YOLO11s'),
    ('ml-debi', '[full]', 'ml-debi YOLOv8'),
    ('lpdnet', '[crop]', 'NVIDIA LPDNet (usa)$^\\dagger$'),
)


def find_latest_matrix_run(root: Path) -> Path | None:
    """Return the most recent finished run whose ``matrix.json`` has a curated set.

    A run is only usable for the headline / cross-dataset tables if it scored the
    curated split (balanced-only runs hold just the public sets).
    """
    best: tuple[str, Path] | None = None
    for matrix_path in root.glob('*/matrix.json'):
        try:
            matrix = json.loads(matrix_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if curated_dataset(matrix) is None:
            continue
        status = matrix_path.parent / 'status.json'
        finished = ''
        if status.is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                finished = str(json.loads(status.read_text()).get('finished_at', ''))
        if best is None or finished > best[0]:
            best = (finished, matrix_path.parent)
    return best[1] if best else None


def curated_dataset(matrix: dict) -> str | None:
    """Identify the curated dataset (the one not in the public benchmark set)."""
    for ds in matrix.get('datasets', []):
        if ds not in PUBLIC_DATASETS:
            return str(ds)
    return None


def model_key(matrix: dict, substr: str, regime: str) -> str | None:
    """Find the matrix model key matching a name substring + regime tag."""
    for name in matrix.get('models', []):
        if substr in name and regime in name:
            return str(name)
    return None


def cell(matrix: dict, model: str | None, dataset: str | None, metric: str) -> float | None:
    if model is None or dataset is None:
        return None
    val = matrix.get('cells', {}).get(model, {}).get(dataset, {}).get(metric)
    return float(val) if isinstance(val, (int, float)) else None


def _fmt(value: float | None, decimals: int = 3) -> str:
    return '--' if value is None else f'{value:.{decimals}f}'


def build_headline_rows(matrix: dict, dataset: str) -> list[str]:
    """Regenerate the ``tab:headline`` body (per-column bolding) for ``dataset``."""
    selected: list[tuple[str, str]] = [
        (display, key)
        for substr, regime, display in HEADLINE_ROWS
        if (key := model_key(matrix, substr, regime)) is not None
    ]
    if not selected:
        return []
    best_acc = {
        f: max(v for _, k in selected if (v := cell(matrix, k, dataset, f)) is not None)
        for f in ACC_FIELDS
    }
    lat = {k: cell(matrix, k, dataset, 'latency_ms') for _, k in selected}
    best_ms = min(v for v in lat.values() if v is not None)

    lines: list[str] = []
    for display, key in selected:
        cells = [display]
        for f in ACC_FIELDS:
            v = cell(matrix, key, dataset, f)
            txt = _fmt(v)
            if v is not None and abs(v - best_acc[f]) < 1e-9:
                txt = f'\\textbf{{{txt}}}'
            cells.append(txt)
        ms_val = lat[key]
        ms = '--' if ms_val is None else str(round(ms_val))
        if ms_val is not None and abs(ms_val - best_ms) < 1e-9:
            ms = f'\\textbf{{{ms}}}'
        cells.append(ms)
        lines.append('        & '.join(cells) + r' \\')
    return lines


def build_cross_dataset_rows(matrix: dict, curated: str) -> list[str]:
    """Regenerate ``tab:matrix`` body: full-frame mAP@.5 per dataset per model."""
    cols = (
        ('ours', 'Ours'),
        ('lpr_nanov11_640', 'lpr_nanov11'),
        ('open-image-models', 'open-img'),
        ('lpdnet', 'LPDNet'),
        ('morsetechlab', 'morsetech'),
    )
    keys = [(label, model_key(matrix, substr, '[full]')) for substr, label in cols]
    order = [curated, *[d for d in PUBLIC_DATASETS if d in matrix.get('datasets', [])]]
    lines: list[str] = []
    for ds in order:
        cells = [ds]
        for _, key in keys:
            cells.append(_fmt(cell(matrix, key, ds, 'map_50')))
        lines.append(' & '.join(cells) + r' \\')
    return lines


def variant_key(matrix: dict, name: str) -> str | None:
    """Resolve a quant-variant model row by exact name, then substring fallback."""
    models = matrix.get('models', [])
    if name in models:
        return name
    for m in models:
        if name in m:
            return str(m)
    return None


def _fmt_delta(value: float | None, decimals: int = 3) -> str:
    """Signed delta, e.g. ``+0.001`` / ``-0.004`` / ``0.000``."""
    if value is None:
        return '--'
    return f'{value:+.{decimals}f}'


def _fp32_baseline_key(matrix: dict) -> str | None:
    """The FP32 reference variants are compared against (ONNX, else the .pt run)."""
    return variant_key(matrix, 'ours_fp32_onnx') or model_key(matrix, 'ours', '[full]')


def build_quant_rows(matrix: dict, dataset: str) -> list[str]:
    """Regenerate the ``tab:quant`` body: size + accuracy + latency per variant.

    Columns: Variant & Size(MB) & mAP@.5:.95 & AP_small & ms & dmAP-vs-FP32. The
    delta is computed (never hand-typed) against the FP32 baseline row.
    """
    present = [(key, label) for name, label in QUANT_VARIANTS if (key := variant_key(matrix, name))]
    if not present:
        return []
    fp32_map = cell(matrix, _fp32_baseline_key(matrix), dataset, 'map_50_95')
    lines: list[str] = []
    for key, label in present:
        mcoco = cell(matrix, key, dataset, 'map_50_95')
        size = cell(matrix, key, dataset, 'size_mb')
        aps = cell(matrix, key, dataset, 'ap_small')
        lat = cell(matrix, key, dataset, 'latency_ms')
        delta = (mcoco - fp32_map) if (mcoco is not None and fp32_map is not None) else None
        ms = '--' if lat is None else str(round(lat))
        cells = [label, _fmt(size, 2), _fmt(mcoco), _fmt(aps), ms, _fmt_delta(delta)]
        lines.append(' & '.join(cells) + r' \\')
    return lines


def build_quant_macros(matrix: dict, dataset: str) -> str:
    """Emit ``\\newcommand`` macros for the quantization prose (size, dmAP, speedup)."""
    fp32_key = _fp32_baseline_key(matrix)
    fp32_map = cell(matrix, fp32_key, dataset, 'map_50_95')
    fp32_size = cell(matrix, fp32_key, dataset, 'size_mb')

    def c(name: str, value: str) -> str:
        return f'\\newcommand{{\\{name}}}{{{value}}}'

    macro_names = {
        'ours_fp16_onnx': 'Fp16',
        'ours_int8_onnx': 'Int8',
        'ours_fp16_coreml': 'CoremlFp16',
        'ours_int8_coreml': 'CoremlInt8',
    }
    out: list[str] = []
    if fp32_size is not None:
        out.append(c('resQuantFp32SizeMB', _fmt(fp32_size, 2)))
    if fp32_map is not None:
        out.append(c('resQuantFp32MapCoco', _fmt(fp32_map)))
    for name, suffix in macro_names.items():
        key = variant_key(matrix, name)
        if key is None:
            continue
        mcoco = cell(matrix, key, dataset, 'map_50_95')
        size = cell(matrix, key, dataset, 'size_mb')
        if mcoco is not None:
            out.append(c(f'resQuant{suffix}MapCoco', _fmt(mcoco)))
        if mcoco is not None and fp32_map is not None:
            out.append(c(f'resQuant{suffix}DeltaMap', _fmt_delta(mcoco - fp32_map)))
        if size is not None:
            out.append(c(f'resQuant{suffix}SizeMB', _fmt(size, 2)))
        if size is not None and fp32_size and size > 0:
            out.append(c(f'resQuant{suffix}SizeReduction', f'{fp32_size / size:.1f}'))
    return ('\n'.join(out) + '\n') if out else ''


def build_macros(matrix: dict, curated: str) -> str:
    """Emit ``\\newcommand`` macros for the inline metrics quoted in the prose."""
    ours_f = model_key(matrix, 'ours', '[full]')
    ours_c = model_key(matrix, 'ours', '[crop]')
    base_keys = [n for n in matrix.get('models', []) if '[full]' in n and 'ours' not in n]
    best_base = max(
        base_keys, key=lambda k: cell(matrix, k, curated, 'map_50_95') or -1.0, default=None
    )

    def c(name: str, value: str) -> str:
        return f'\\newcommand{{\\{name}}}{{{value}}}'

    out = ['% Auto-generated by scripts/curation/bakeoff/paper_numbers.py -- do not hand-edit.']
    out += [
        c('resOursMapCoco', _fmt(cell(matrix, ours_f, curated, 'map_50_95'))),
        c('resOursMapFifty', _fmt(cell(matrix, ours_f, curated, 'map_50'))),
        c('resOursMeanIoU', _fmt(cell(matrix, ours_f, curated, 'mean_iou'))),
        c('resOursMapCocoCrop', _fmt(cell(matrix, ours_c, curated, 'map_50_95'))),
        c('resBaselineMapCoco', _fmt(cell(matrix, best_base, curated, 'map_50_95'))),
        c('resBaselineMeanIoU', _fmt(cell(matrix, best_base, curated, 'mean_iou'))),
    ]
    lat = cell(matrix, ours_f, curated, 'latency_ms')
    out.append(c('resOursLatency', '--' if lat is None else str(round(lat))))
    return '\n'.join(out) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bakeoff-root', type=Path, default=DEFAULT_BAKEOFF_ROOT)
    ap.add_argument('--run-dir', type=Path, default=None, help='Matrix run dir (else latest).')
    ap.add_argument('--out', type=Path, default=None, help='Write numbers.tex macros here.')
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_matrix_run(args.bakeoff_root)
    if run_dir is None or not (run_dir / 'matrix.json').is_file():
        raise SystemExit(f'no matrix.json run found under {args.bakeoff_root}')
    matrix = json.loads((run_dir / 'matrix.json').read_text())
    curated = curated_dataset(matrix)
    if curated is None:
        raise SystemExit(f'could not identify curated dataset in {run_dir}/matrix.json')
    logger.info('Run: %s   curated dataset: %s', run_dir, curated)

    logger.info('\n=== tab:headline body (curated split) ===')
    for line in build_headline_rows(matrix, curated):
        logger.info(line)
    logger.info('\n=== tab:matrix body (full-frame mAP@.5 by dataset) ===')
    for line in build_cross_dataset_rows(matrix, curated):
        logger.info(line)

    quant_rows = build_quant_rows(matrix, curated)
    if quant_rows:
        logger.info('\n=== tab:quant body (size / accuracy / latency by precision) ===')
        for line in quant_rows:
            logger.info(line)

    macros = build_macros(matrix, curated)
    quant_macros = build_quant_macros(matrix, curated)
    if quant_macros:
        macros = macros + quant_macros
    logger.info('\n=== inline-metric macros ===\n%s', macros)

    if args.out:
        args.out.write_text(macros)
        logger.info('Wrote macros -> %s', args.out)


if __name__ == '__main__':
    main()

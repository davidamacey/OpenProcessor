"""On-demand bake-off job runner (entrypoint for the bake-off evaluator container).

Reads a job spec describing one-or-more datasets + a list of models, scores
every model on every dataset through the harness (each in its own subprocess so
one crash doesn't abort the job), verifies each frozen test set first, then
aggregates into a model x dataset ``matrix.json`` (with per-dataset best flags)
plus per-dataset ``comparison.json``. Writes a ``status.json`` the API serves to
the UI.

Modes:
    --watch /eval_jobs   poll for ``*.job.json`` (like kb_trainer)
    --job <file>         run one job spec and exit

Job spec (JSON) --- ``datasets`` is the matrix form; ``dataset`` (singular) is
still accepted for back-compat::

    {
        'job_id': '2026-05-25T10-00',
        'datasets': [
            {'name': 'curated', 'path': '/data/lpr_exports/<run>'},
            {'name': 'andrewmvd', 'path': '/mnt/nvm/datasets/plates/andrewmvd_car_plate'},
        ],
        'verify_frozen': true,
        'out_dir': '/data/bakeoff/<job_id>',
        'models': [
            {
                'backend': 'ultralytics',
                'weights': '/runs/.../best.pt',
                'name': 'ours-yolo26',
                'imgsz': 640,
                'mode': 'both',
                'device': 'cuda',
            },
            {'backend': 'open-image-models', 'name': 'open-image-models', 'mode': 'both'},
        ],
    }
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compare import build_comparison, to_markdown
from .freeze import verify


# GPU pool + concurrency for parallel scoring. The detectors are tiny (hundreds
# of MB), so a single A6000 hosts several at once; image loading is the real
# bottleneck, parallelized across CPU cores. Tune on the evaluator via env vars
# OP_BAKEOFF_GPUS (comma-separated PCI ids; default the A6000 with headroom) and
# OP_BAKEOFF_CONCURRENCY (parallel task slots).
_GPUS: list[str] = [
    g.strip() for g in os.environ.get('OP_BAKEOFF_GPUS', '0').split(',') if g.strip()
]
_CONCURRENCY: int = max(1, int(os.environ.get('OP_BAKEOFF_CONCURRENCY', '4')))


_OPT_FLAGS = {
    'weights': '--weights',
    'imgsz': '--imgsz',
    'device': '--device',
    'plate_class_id': '--plate-class-id',
    'mode': '--mode',
    'lpdnet_variant': '--lpdnet-variant',
    'triton_url': '--triton-url',
    'triton_model': '--triton-model',
    'vehicle_weights': '--vehicle-weights',
    'vehicle_classes': '--vehicle-classes',
    'vehicle_imgsz': '--vehicle-imgsz',
    'lpr_backend': '--lpr-backend',
    'lpr_imgsz': '--lpr-imgsz',
    'training_data': '--training-data',
    'ort_providers': '--ort-providers',
    'coords_normalized': '--coords-normalized',
    'coreml_compute_units': '--coreml-compute-units',
}

# Metrics carried into the matrix cells. "Higher is better" except latency_ms
# and size_mb (both lower-is-better — the lighter/faster axis).
_CELL_METRICS = (
    'map_50',
    'map_50_95',
    'ap_small',
    'mean_iou',
    'precision',
    'recall',
    'f1',
    'latency_ms',
    'size_mb',
)

# Metrics where the winning (bolded) cell is the minimum, not the maximum.
_LOWER_IS_BETTER = {'latency_ms', 'size_mb'}


def _quantize_and_variant_models(
    quant: dict[str, Any], datasets: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Run export/quantize.py for a checkpoint, return the variant model specs.

    Lets a single bake-off job be fully seamless: export the trained YOLO26 model
    to portable ONNX (fp32/fp16/int8), then score those variants alongside the
    other models so the matrix + frontend QuantizationPanel show size / speed /
    accuracy with no manual step. The ``quantize`` block:

        {"model_id": "ours_yolo26n", "checkpoint": "/runs/.../best.pt",
         "formats": ["fp32_onnx","fp16_onnx","int8_onnx"], "n_calib": 1000,
         "calib_dataset": "/data/lpr_exports/<run>",  # default: first dataset
         "out_root": "/data/quant"}
    """
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from export.quantize import run as quantize_run

    model_id = quant.get('model_id', 'ours_yolo26n')
    formats = quant.get('formats') or ['fp32_onnx', 'fp16_onnx', 'int8_onnx']
    out_root = Path(quant.get('out_root', '/data/quant'))
    calib = quant.get('calib_dataset') or (datasets[0]['path'] if datasets else None)
    checkpoint = quant.get('checkpoint')
    print(f'[bakeoff] quantize: exporting {model_id} {formats} from {checkpoint}', flush=True)
    quantize_run(
        model_id,
        formats,
        out_root,
        pt_override=Path(checkpoint) if checkpoint else None,
        calib_override=Path(calib) if calib else None,
        n_calib_override=quant.get('n_calib'),
    )

    qdir = out_root / model_id
    # fp32/fp16 run on the GPU (portable, no engine build); INT8 QDQ runs on the
    # CPU EP (reliable; the GPU INT8 path needs the TensorRT EP) — its latency is
    # the edge/CPU datapoint while its mAP is EP-independent.
    variant_specs = [
        ('fp32_onnx', 'fp32.onnx', 'CUDAExecutionProvider,CPUExecutionProvider', 'FP32 ONNX'),
        (
            'fp16_onnx',
            'fp16.onnx',
            'CUDAExecutionProvider,CPUExecutionProvider',
            'FP16 ONNX (NVIDIA)',
        ),
        ('int8_onnx', 'int8_qdq.onnx', 'CPUExecutionProvider', 'INT8 QDQ ONNX'),
    ]
    out: list[dict[str, Any]] = []
    for fmt, fname, providers, label in variant_specs:
        if fmt not in formats:
            continue
        weights = qdir / fname
        if not weights.is_file():
            print(f'[bakeoff] quantize: missing {weights}, skipping', flush=True)
            continue
        out.append(
            {
                'backend': 'onnxruntime',
                'name': f'ours_{fmt.replace("_onnx", "")}_onnx',
                'weights': str(weights),
                'imgsz': quant.get('imgsz', 640),
                'device': 'cuda',
                'coords_normalized': False,
                'ort_providers': providers,
                'mode': 'full',
                'training_data': label,
            }
        )
    return out


def _run_throughput_sweep(
    quant: dict[str, Any], datasets: list[dict[str, str]], out_dir: Path
) -> None:
    """Steady-state img/s for every exported variant on CPU + GPU (best-effort).

    Runs ``throughput.py`` per variant/EP so the auto pipeline captures model
    *speed* (not just accuracy) -- the JSONs land in ``<out_dir>/throughput/``.
    """
    model_id = quant.get('model_id', 'ours_yolo26n')
    qdir = Path(quant.get('out_root', '/data/quant')) / model_id
    imgsz = str(quant.get('imgsz', 640))
    if not datasets:
        return
    images = Path(datasets[0]['path']) / 'images' / 'test'
    if not images.is_dir():
        print(f'[bakeoff] throughput: images dir {images} missing, skipping', flush=True)
        return
    tput_out = out_dir / 'throughput'
    # Each entry: backend, weights filename, label, ORT providers.
    runs = [
        ('ultralytics', 'source.pt', 'pt_gpu', None),
        ('onnxruntime', 'fp32.onnx', 'fp32_cuda', 'CUDAExecutionProvider,CPUExecutionProvider'),
        ('onnxruntime', 'fp32.onnx', 'fp32_cpu', 'CPUExecutionProvider'),
        ('onnxruntime', 'fp16.onnx', 'fp16_cuda', 'CUDAExecutionProvider,CPUExecutionProvider'),
        ('onnxruntime', 'fp16.onnx', 'fp16_cpu', 'CPUExecutionProvider'),
        ('onnxruntime', 'int8_qdq.onnx', 'int8_cuda', 'CUDAExecutionProvider,CPUExecutionProvider'),
        ('onnxruntime', 'int8_qdq.onnx', 'int8_cpu', 'CPUExecutionProvider'),
    ]
    for backend, fname, label, provs in runs:
        weights = qdir / fname
        if not weights.is_file():
            continue
        argv = [
            sys.executable,
            '-m',
            'scripts.curation.bakeoff.throughput',
            '--backend',
            backend,
            '--weights',
            str(weights),
            '--label',
            label,
            '--images',
            str(images),
            '--imgsz',
            imgsz,
            '--n-images',
            '200',
            '--warmup',
            '30',
            '--min-seconds',
            '6',
            '--out',
            str(tput_out),
        ]
        if backend == 'onnxruntime' and provs:
            argv += ['--ort-providers', provs]
        if backend == 'ultralytics':
            argv += ['--device', '0']
        try:
            subprocess.run(argv, check=True)
        except subprocess.CalledProcessError as exc:
            print(f'[bakeoff] throughput {label} failed: {exc}', flush=True)


def _run_coreml_mac_leg(quant: dict[str, Any], out_dir: Path) -> None:
    """Drive the Mac Studio CoreML export+bench over SSH (best-effort, opt-in).

    Gated by the ``OP_COREML_HOST`` env var (e.g. ``superstudio@superstudio.home.arpa``)
    so it's a safe no-op until the evaluator is provisioned with ssh/rsync + a key
    that can reach the Mac on the LAN. Exports FP16/INT8 CoreML on the Mac, benchmarks
    ANE+CPU, and pulls the .mlpackage + throughput JSON back.
    """
    host = os.environ.get('OP_COREML_HOST')
    if not host:
        print('[bakeoff] coreml: OP_COREML_HOST not set, skipping Mac leg', flush=True)
        return
    repo_root = Path(__file__).resolve().parents[3]
    model_id = quant.get('model_id', 'ours_yolo26n')
    argv = [
        sys.executable,
        str(repo_root / 'export' / 'quantize_coreml_mac.py'),
        '--model',
        model_id,
        '--host',
        host,
        '--pull-into',
        str(out_dir / 'coreml'),
    ]
    print(f'[bakeoff] coreml: driving Mac leg on {host} ...', flush=True)
    try:
        subprocess.run(argv, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(
            f'[bakeoff] coreml: Mac leg failed (need ssh/rsync + key in image?): {exc}', flush=True
        )


def _expand_modes(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand a model with ``mode: both`` into separate full + crop runs."""
    out: list[dict[str, Any]] = []
    for m in models:
        if m.get('mode') == 'both':
            base = m.get('name', m['backend'])
            for mode in ('full', 'crop'):
                c = dict(m)
                c['mode'] = mode
                c['name'] = f'{base} [{mode}]'
                out.append(c)
        else:
            out.append(m)
    return out


def _dataset_specs(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize the job spec to a list of ``{name, path}`` datasets.

    Accepts the matrix form (``datasets``: list of dicts or path strings) or the
    legacy single ``dataset`` path.
    """
    out: list[dict[str, str]] = []
    if spec.get('datasets'):
        for d in spec['datasets']:
            if isinstance(d, str):
                out.append({'name': Path(d).name, 'path': d})
            else:
                path = str(d['path'])
                out.append({'name': str(d.get('name') or Path(path).name), 'path': path})
    elif spec.get('dataset'):
        ds = str(spec['dataset'])
        out.append({'name': Path(ds).name, 'path': ds})
    return out


def _model_argv(dataset: Path, out_dir: Path, model: dict[str, Any]) -> list[str]:
    argv = [
        sys.executable,
        '-m',
        'scripts.curation.bakeoff.run',
        '--dataset',
        str(dataset),
        '--backend',
        model['backend'],
        '--name',
        model.get('name', model['backend']),
        '--out-dir',
        str(out_dir),
    ]
    # Per-cluster stratum metrics only exist for our own exports; public sets
    # have no stratum_map.json, so pass it only when present.
    stratum_map = dataset / 'stratum_map.json'
    if stratum_map.is_file():
        argv += ['--stratum-map', str(stratum_map)]
    for key, flag in _OPT_FLAGS.items():
        if key in model and model[key] is not None:
            argv += [flag, str(model[key])]
    return argv


def _build_matrix(
    dataset_names: list[str], per_dataset: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Assemble a model x dataset matrix with per-(dataset,metric) best flags.

    ``cells[model][dataset]`` holds the metric dict; ``best[dataset][metric]`` is
    the winning model name (max, or min for ``latency_ms``) so the UI/paper can
    bold it.
    """
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    model_order: list[str] = []
    for ds_name in dataset_names:
        comp = per_dataset.get(ds_name)
        if not comp:
            continue
        for row in comp.get('models', []):
            mname = row['model']
            if mname not in cells:
                cells[mname] = {}
                model_order.append(mname)
            cells[mname][ds_name] = {k: row.get(k) for k in _CELL_METRICS}

    best: dict[str, dict[str, str]] = {ds: {} for ds in dataset_names}
    for ds in dataset_names:
        for metric in _CELL_METRICS:
            scored = [
                (m, cells[m][ds][metric])
                for m in cells
                if ds in cells[m] and isinstance(cells[m][ds].get(metric), (int, float))
            ]
            if not scored:
                continue
            winner = (
                min(scored, key=lambda x: x[1])
                if metric in _LOWER_IS_BETTER
                else max(scored, key=lambda x: x[1])
            )
            best[ds][metric] = winner[0]

    return {
        'datasets': dataset_names,
        'models': model_order,
        'metrics': list(_CELL_METRICS),
        'cells': cells,
        'best': best,
    }


def _run_task(
    ds_path: Path, ds_out: Path, model: dict[str, Any], gpu: str
) -> tuple[str, bool, str | None]:
    """Score one model on one dataset, pinned to ``gpu`` (PCI id).

    Forces ``--device cuda`` + ``CUDA_VISIBLE_DEVICES=<gpu>`` so Ultralytics
    can't rewrite the pin to a physical index, letting many tasks run in
    parallel each owning one GPU from the pool. Triton-backed models ignore the
    pin (they call the Triton server).
    """
    name = model.get('name', model['backend'])
    pinned = {**model, 'device': 'cuda'}
    env = dict(os.environ)
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    try:
        subprocess.run(_model_argv(ds_path, ds_out, pinned), check=True, env=env)
        return name, True, None
    except subprocess.CalledProcessError as exc:
        return name, False, str(exc)


def _write_status(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'status.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


def run_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one bake-off job spec (single dataset or matrix); return status."""
    job_id = spec.get('job_id', datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    out_dir = Path(spec.get('out_dir', f'/data/bakeoff/{job_id}'))
    datasets = _dataset_specs(spec)
    raw_models = list(spec.get('models', []))
    # Optional: export the trained model to portable ONNX first, then score those
    # quantized variants in this same job (seamless export -> benchmark -> matrix).
    if spec.get('quantize'):
        try:
            raw_models += _quantize_and_variant_models(spec['quantize'], datasets)
        except Exception as exc:  # don't abort the whole job if export fails
            print(f'[bakeoff] quantize step failed: {exc}', flush=True)
    models = _expand_modes(raw_models)
    verify_frozen = spec.get('verify_frozen', True)
    model_names = [m.get('name', m['backend']) for m in models]

    status: dict[str, Any] = {
        'job_id': job_id,
        'state': 'running',
        'datasets': [d['name'] for d in datasets],
        'dataset': datasets[0]['path'] if datasets else None,  # legacy field
        'started_at': datetime.now(UTC).isoformat(),
        'models': model_names,
        'completed': [],
        'failed': [],
        'progress': {'done': 0, 'total': len(datasets) * len(models)},
    }
    _write_status(out_dir, status)

    if not datasets:
        status.update(state='error', error='no datasets in job spec')
        _write_status(out_dir, status)
        return status

    total = len(datasets) * len(models)
    done = 0

    # Verify frozen sets up front; collect the datasets we'll actually score.
    valid: list[tuple[str, Path, Path]] = []
    for ds in datasets:
        ds_path = Path(ds['path'])
        ds_name = ds['name']
        if verify_frozen:
            ok, msg = verify(ds_path)
            if not ok:
                status['failed'].append(
                    {'dataset': ds_name, 'error': f'frozen verify failed: {msg}'}
                )
                done += len(models)
                continue
        valid.append((ds_name, ds_path, out_dir / ds_name))
    status['progress'] = {'done': done, 'total': total}
    status['gpus'] = _GPUS
    status['concurrency'] = _CONCURRENCY
    _write_status(out_dir, status)

    # Run the (dataset x model) grid in parallel across the GPU pool. Each task
    # is pinned to one GPU (round-robin); tiny detectors + parallel image I/O
    # turn a long sequential sweep into a saturated one.
    gpus = itertools.cycle(_GPUS)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        futs = {}
        for ds_name, ds_path, ds_out in valid:
            for model in models:
                futs[pool.submit(_run_task, ds_path, ds_out, model, next(gpus))] = (
                    ds_name,
                    model.get('name', model['backend']),
                )
        for fut in as_completed(futs):
            ds_name, _mname = futs[fut]
            name, ok, err = fut.result()
            with lock:
                if ok:
                    status['completed'].append(f'{ds_name}:{name}')
                else:
                    status['failed'].append({'dataset': ds_name, 'model': name, 'error': err})
                done += 1
                status['progress'] = {'done': done, 'total': total}
                _write_status(out_dir, status)

    # Aggregate each dataset's per-model JSON into its comparison.
    per_dataset: dict[str, dict[str, Any]] = {}
    for ds_name, _ds_path, ds_out in valid:
        comp = build_comparison(ds_out)
        (ds_out / 'comparison.json').write_text(json.dumps(comp, indent=2), encoding='utf-8')
        (ds_out / 'comparison.md').write_text(to_markdown(comp), encoding='utf-8')
        per_dataset[ds_name] = comp

    matrix = _build_matrix([d['name'] for d in datasets], per_dataset)
    (out_dir / 'matrix.json').write_text(json.dumps(matrix, indent=2), encoding='utf-8')

    # Optional: steady-state throughput (img/s, CPU+GPU) for the quant variants.
    if spec.get('quantize', {}).get('throughput'):
        try:
            _run_throughput_sweep(spec['quantize'], datasets, out_dir)
        except Exception as exc:  # best-effort; never fail the whole job
            print(f'[bakeoff] throughput sweep failed: {exc}', flush=True)

    # Optional: Mac Studio CoreML export + ANE/CPU benchmark over the LAN.
    if spec.get('quantize', {}).get('coreml'):
        try:
            _run_coreml_mac_leg(spec['quantize'], out_dir)
        except Exception as exc:  # best-effort
            print(f'[bakeoff] coreml mac leg failed: {exc}', flush=True)

    # Back-compat: a top-level comparison.json (the first/primary dataset) so the
    # existing single-dataset results view keeps working.
    primary = datasets[0]['name']
    if primary in per_dataset:
        (out_dir / 'comparison.json').write_text(
            json.dumps(per_dataset[primary], indent=2), encoding='utf-8'
        )
        (out_dir / 'comparison.md').write_text(to_markdown(per_dataset[primary]), encoding='utf-8')

    status.update(state='done', finished_at=datetime.now(UTC).isoformat(), matrix=matrix)
    _write_status(out_dir, status)
    return status


def _watch(jobs_dir: Path, poll: float) -> int:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    done_dir = jobs_dir / 'done'
    done_dir.mkdir(exist_ok=True)
    print(f'bakeoff_runner watching {jobs_dir} (poll {poll}s)', flush=True)
    while True:
        for job_file in sorted(jobs_dir.glob('*.job.json')):
            try:
                spec = json.loads(job_file.read_text(encoding='utf-8'))
                print(f'running job {job_file.name}', flush=True)
                run_job(spec)
            except Exception as exc:  # keep the watcher alive on a bad job
                print(f'job {job_file.name} failed: {exc}', flush=True)
            job_file.rename(done_dir / job_file.name)
        time.sleep(poll)


def main() -> int:
    p = argparse.ArgumentParser(description='Bake-off job runner (evaluator container).')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--watch', type=Path, help='Watch a dir for *.job.json')
    g.add_argument('--job', type=Path, help='Run a single job spec and exit')
    p.add_argument('--poll', type=float, default=5.0)
    args = p.parse_args()
    if args.watch:
        return _watch(args.watch, args.poll)
    spec = json.loads(args.job.read_text(encoding='utf-8'))
    status = run_job(spec)
    print(json.dumps(status.get('matrix', {}), indent=2))
    return 0 if status.get('state') == 'done' else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""On-demand bake-off job runner (entrypoint for the bake-off evaluator container).

Reads a job spec (schema v2) describing one-or-more eval datasets + a list of
models, scores every model on every dataset through the harness (each in its
own subprocess so one crash doesn't abort the job), after checking that each
dataset's test split is still the one the job was enqueued against. Then
aggregates each dataset's per-model reports into ``<dir_name>/comparison.json``
and all datasets into a model x dataset ``matrix.json``, and keeps a
``status.json`` the API serves to the UI.

Modes:
    --watch /eval_jobs   poll for ``*.job.json`` (same protocol as the training job runner)
    --job <file>         run one job spec and exit

Job spec v2 (written by the API)::

    {
        'schema_version': 2,
        'job_id': '20260925T010203Z',
        'profile': 'generic',  # registered name or profile .json path
        'out_dir': '/var/lib/openprocessor/bakeoff_out/<job_id>',
        'datasets': [
            {
                'id': 'export:20260924T233203Z',
                'dir_name': 'export__20260924T233203Z',
                'path': '/exports/20260924T233203Z',
                'test_label_sha': '<16 hex>',  # verified before scoring
                'frozen_test_sha': '<16 hex>',
                'eval_class_ids': [37, 38, 43, 51, 78],
            }
        ],
        'models': [
            {
                'model': 'run:<run_id>',  # unique key (report stem, matrix row)
                'display_name': '<run_id>',
                'source': 'run',  # run | baseline | custom
                'run_id': '<run_id>',
                'backend': 'ultralytics',
                'weights': '/.../weights/best.pt',
                'imgsz': 640,
                'mode': 'full',  # full | crop | both
                'backend_options': {},
                'triton_model': None,
                'training_data': None,
                'class_map_by_dataset': {'export:...': {'0': 37}},  # null = match names
                'train_test_overlap_by_dataset': {
                    'export:...': {'n_images': 0, 'fraction': 0.0}
                },
            }
        ],
        'quantize': None,  # or the block documented in quant_stage._quantize_and_variant_models
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

from . import quant_stage
from .compare import build_comparison, build_matrix, to_markdown
from .freeze import test_sha
from .profile import resolve_profile
from .quant_stage import COREML_UNAVAILABLE


SCHEMA_VERSION = 2

# GPU pool + concurrency for parallel scoring. The detectors are tiny (hundreds
# of MB), so one GPU hosts several at once; image loading is the real
# bottleneck, parallelized across CPU cores. Tune on the evaluator via env vars
# OP_BAKEOFF_GPUS (comma-separated container-local ids) and
# OP_BAKEOFF_CONCURRENCY (parallel task slots).
_GPUS: list[str] = [
    g.strip() for g in os.environ.get('OP_BAKEOFF_GPUS', '0').split(',') if g.strip()
]
_CONCURRENCY: int = max(1, int(os.environ.get('OP_BAKEOFF_CONCURRENCY', '4')))

# Job-spec model field -> run.py flag (scalar values, passed when not None).
_OPT_FLAGS = {
    'display_name': '--display-name',
    'source': '--source',
    'run_id': '--run-id',
    'weights': '--weights',
    'imgsz': '--imgsz',
    'device': '--device',
    'mode': '--mode',
    'triton_url': '--triton-url',
    'triton_model': '--triton-model',
    'training_data': '--training-data',
}


def _expand_modes(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand a model with ``mode: both`` into separate full + crop runs."""
    out: list[dict[str, Any]] = []
    for m in models:
        if m.get('mode') == 'both':
            for mode in ('full', 'crop'):
                c = dict(m)
                c['mode'] = mode
                c['model'] = f'{m["model"]}:{mode}'
                c['display_name'] = f'{m.get("display_name") or m["model"]} [{mode}]'
                out.append(c)
        else:
            out.append(m)
    return out


def _model_argv(dataset: Path, out_dir: Path, ds_id: str, model: dict[str, Any]) -> list[str]:
    """The ``run.py`` command line scoring ``model`` on one dataset."""
    argv = [
        sys.executable,
        '-m',
        'scripts.curation.bakeoff.run',
        '--dataset',
        str(dataset),
        '--backend',
        model['backend'],
        '--model-key',
        model['model'],
        '--out-dir',
        str(out_dir),
        '--class-map-json',
        json.dumps((model.get('class_map_by_dataset') or {}).get(ds_id)),
        '--backend-options-json',
        json.dumps(model.get('backend_options') or {}),
        '--train-test-overlap-json',
        json.dumps((model.get('train_test_overlap_by_dataset') or {}).get(ds_id)),
    ]
    if model.get('profile'):
        argv += ['--profile', str(model['profile'])]
    # Per-cluster stratum metrics only exist for exports that ship a
    # stratum_map.json; pass it only when present.
    stratum_map = dataset / 'stratum_map.json'
    if stratum_map.is_file():
        argv += ['--stratum-map', str(stratum_map)]
    for key, flag in _OPT_FLAGS.items():
        if model.get(key) is not None:
            argv += [flag, str(model[key])]
    return argv


def _run_task(
    ds_path: Path, ds_out: Path, ds_id: str, model: dict[str, Any], gpu: str
) -> tuple[str, bool, str | None]:
    """Score one model on one dataset, pinned to ``gpu`` (container-local id).

    Forces ``--device cuda`` + ``CUDA_VISIBLE_DEVICES=<gpu>`` so Ultralytics
    can't rewrite the pin to a physical index, letting many tasks run in
    parallel each owning one GPU from the pool. Triton-backed models ignore the
    pin (they call the Triton server).
    """
    pinned = {**model, 'device': 'cuda'}
    env = dict(os.environ)
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    try:
        subprocess.run(_model_argv(ds_path, ds_out, ds_id, pinned), check=True, env=env)
        return model['model'], True, None
    except subprocess.CalledProcessError as exc:
        return model['model'], False, str(exc)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(path)


def _failure(
    error: str, *, stage: str | None = None, dataset: str | None = None, model: str | None = None
) -> dict[str, Any]:
    return {'stage': stage, 'dataset': dataset, 'model': model, 'error': error}


def _error_status(job_id: str, out_dir: Path, error: str) -> dict[str, Any]:
    status: dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'job_id': job_id,
        'state': 'error',
        'profile': None,
        'datasets': [],
        'models': [],
        'started_at': datetime.now(UTC).isoformat(),
        'finished_at': datetime.now(UTC).isoformat(),
        'progress': {'done': 0, 'total': 0},
        'completed': [],
        'failed': [],
        'error': error,
    }
    _write_json(out_dir / 'status.json', status)
    return status


def _check_test_splits(
    datasets: list[dict[str, Any]], failed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Datasets whose test labels still hash to the enqueue-time ``test_label_sha``."""
    valid: list[dict[str, Any]] = []
    for ds in datasets:
        now = test_sha(Path(ds['path']))[0]
        expected = ds.get('test_label_sha')
        if now != expected:
            failed.append(
                _failure(
                    f'test split changed since enqueue: expected {expected}, now {now}',
                    dataset=ds['id'],
                )
            )
            continue
        valid.append(ds)
    return valid


def run_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one job spec v2; return (and write) its final status."""
    job_id = str(spec.get('job_id') or datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    out_dir = Path(spec.get('out_dir') or f'/data/bakeoff/{job_id}')
    if spec.get('schema_version') != SCHEMA_VERSION:
        return _error_status(
            job_id,
            out_dir,
            f'unsupported bake-off job spec (schema_version != {SCHEMA_VERSION}); '
            're-submit it through POST /bakeoff/run',
        )
    job_profile = spec.get('profile')
    try:
        profile = resolve_profile(job_profile)
    except ValueError as exc:
        return _error_status(job_id, out_dir, str(exc))

    datasets: list[dict[str, Any]] = list(spec.get('datasets') or [])
    raw_models: list[dict[str, Any]] = list(spec.get('models') or [])
    quant = spec.get('quantize') or {}
    failed: list[dict[str, Any]] = []
    # Optional: export the trained model to portable ONNX first, then score those
    # quantized variants in this same job (seamless export -> benchmark -> matrix).
    if quant:
        try:
            variants = quant_stage._quantize_and_variant_models(quant, datasets, out_dir)
        except Exception as exc:  # recorded below; scoring of the other models goes on
            print(f'[bakeoff] quantize step failed: {exc}', flush=True)
            failed.append(_failure(f'{type(exc).__name__}: {exc}', stage='quantize'))
        else:
            overlap = next(
                (
                    m.get('train_test_overlap_by_dataset')
                    for m in raw_models
                    if m['model'] == quant.get('model_key_prefix')
                ),
                None,
            )
            for v in variants:
                v['train_test_overlap_by_dataset'] = dict(overlap or {})
            raw_models += variants
    if quant.get('coreml'):
        failed.append(_failure(COREML_UNAVAILABLE, stage='coreml'))
    if job_profile:
        raw_models = [{**m, 'profile': job_profile} for m in raw_models]
    models = _expand_modes(raw_models)

    total = len(datasets) * len(models)
    status: dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'job_id': job_id,
        'state': 'running',
        'profile': profile.name,
        'datasets': [d['id'] for d in datasets],
        'models': [m['model'] for m in models],
        'started_at': datetime.now(UTC).isoformat(),
        'finished_at': None,
        'progress': {'done': 0, 'total': total},
        'completed': [],
        'failed': failed,
        'error': None,
    }
    _write_json(out_dir / 'status.json', status)

    def finish(state: str, error: str | None = None) -> dict[str, Any]:
        status.update(state=state, error=error, finished_at=datetime.now(UTC).isoformat())
        _write_json(out_dir / 'status.json', status)
        return status

    if not models:
        # e.g. a quantize-only job whose export failed: nothing left to score.
        reason = '; '.join(f'{f["stage"]}: {f["error"]}' for f in failed if f['stage'])
        return finish('error', reason or 'no models to score')
    if not datasets:
        return finish('error', 'no datasets in job spec')

    valid = _check_test_splits(datasets, failed)
    done = (len(datasets) - len(valid)) * len(models)
    status['progress'] = {'done': done, 'total': total}
    _write_json(out_dir / 'status.json', status)
    if not valid:
        return finish('error', 'no dataset could be scored: ' + failed[-1]['error'])

    # Run the (dataset x model) grid in parallel across the GPU pool. Each task
    # is pinned to one GPU (round-robin); tiny detectors + parallel image I/O
    # turn a long sequential sweep into a saturated one.
    gpus = itertools.cycle(_GPUS)
    lock = threading.Lock()
    model_failures: dict[str, list[dict[str, Any]]] = {d['id']: [] for d in valid}
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        futs = {
            pool.submit(
                _run_task,
                Path(ds['path']),
                out_dir / ds['dir_name'],
                ds['id'],
                model,
                next(gpus),
            ): ds['id']
            for ds in valid
            for model in models
        }
        for fut in as_completed(futs):
            ds_id = futs[fut]
            key, ok, err = fut.result()
            with lock:
                if ok:
                    status['completed'].append({'dataset': ds_id, 'model': key})
                else:
                    status['failed'].append(_failure(str(err), dataset=ds_id, model=key))
                    model_failures[ds_id].append({'model': key, 'error': err})
                done += 1
                status['progress'] = {'done': done, 'total': total}
                _write_json(out_dir / 'status.json', status)

    # Aggregate each dataset's per-model reports into its comparison.
    thresholds = {
        'conf_floor': profile.conf_floor,
        'nms_iou': profile.nms_iou,
        'op_conf': profile.op_conf,
        'op_iou': profile.op_iou,
    }
    per_dataset: dict[str, dict[str, Any]] = {}
    for ds in valid:
        ds_out = out_dir / ds['dir_name']
        ds_out.mkdir(parents=True, exist_ok=True)
        comp = build_comparison(
            ds_out,
            rank_by=profile.rank_metric,
            dataset_meta={
                'id': ds['id'],
                'frozen_test_sha': ds.get('frozen_test_sha'),
                'test_label_sha': ds.get('test_label_sha'),
            },
            job_id=job_id,
            profile=profile.name,
            thresholds=thresholds,
            failed=model_failures[ds['id']],
        )
        _write_json(ds_out / 'comparison.json', comp)
        (ds_out / 'comparison.md').write_text(to_markdown(comp), encoding='utf-8')
        per_dataset[ds['id']] = comp

    matrix = build_matrix(datasets, per_dataset, job_id=job_id, rank_by=profile.rank_metric)
    _write_json(out_dir / 'matrix.json', matrix)

    # Optional: steady-state throughput (img/s, CPU+GPU) for the quant variants.
    if quant.get('throughput'):
        try:
            quant_stage._run_throughput_sweep(quant, datasets, out_dir)
        except Exception as exc:  # best-effort; recorded, never fails the whole job
            print(f'[bakeoff] throughput sweep failed: {exc}', flush=True)
            status['failed'].append(_failure(str(exc), stage='throughput'))

    return finish('done')


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
    # The evaluator image ships src/ next to scripts/ (PYTHONPATH=/app).
    from src.config.retired_env import reject_retired_env

    reject_retired_env()
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
    print(json.dumps(status, indent=2))
    return 0 if status.get('state') == 'done' else 1


__all__ = ['COREML_UNAVAILABLE', 'main', 'run_job']


if __name__ == '__main__':
    raise SystemExit(main())

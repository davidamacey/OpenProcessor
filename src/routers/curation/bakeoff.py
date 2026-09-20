"""Bake-off API — trigger model-comparison runs and serve their results.

yolo-api pins an older Ultralytics release and cannot load every model
family the harness benchmarks, so it does NOT run the bake-off itself.
Instead it writes a ``<job_id>.job.json`` into a shared dir that an
on-demand evaluator container watches (same pattern as the training
router). The evaluator scores every model on the frozen test split, logs
to MLflow, and writes ``status.json`` + ``comparison.json`` back here for
the UI to read.

Endpoints:
    POST /bakeoff/run            enqueue a comparison job
    GET  /bakeoff/runs           list past/current runs
    GET  /bakeoff/status/{id}    job progress (running/done/error)
    GET  /bakeoff/results/{id}   ranked comparison rows for the UI

Ported from a private reference vehicle/license-plate curation stack's
bake-off router (see ``docs/design/curation_design_rationale.md`` for
the genericization rationale). The bake-off harness itself lives at
``scripts/curation/bakeoff/`` (not under ``src/`` — see that package's
module docstring for why).
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.config import get_gpu_arbiter_config
from src.core.logging import get_logger
from src.routers.curation._common import router


logger = get_logger(__name__)


# Both yolo-api and the bake-off evaluator container mount the training-data
# root at the same absolute path, so these are shared between the two
# containers. `LEGACY_BAKEOFF_*` env var names are unchanged from the reference
# implementation (established precedent elsewhere in this port — only the
# hardcoded default *values* are genericized).
#
# JOBS_DIR defaults to GpuArbiterConfig.bakeoff_jobs_dir when configured, so
# gpu_arbiter.bakeoff_active() (the reconcile loop's "is a bake-off queued or
# running" check) watches the SAME directory this router writes job.json
# into, without requiring the operator to set the LEGACY_BAKEOFF_JOBS_DIR env var
# and the GpuArbiterConfig field to the same value independently.
_configured_jobs_dir = get_gpu_arbiter_config().bakeoff_jobs_dir
JOBS_DIR = Path(
    os.environ.get(
        'LEGACY_BAKEOFF_JOBS_DIR',
        _configured_jobs_dir or '/data/curation_train_data/bakeoff_jobs',
    )
)
OUT_DIR = Path(os.environ.get('LEGACY_BAKEOFF_OUT_DIR', '/data/curation_train_data/bakeoff'))
# Training runs land here. The trainer records checkpoint_path as a /runs/...
# container path; the evaluator container mounts the same dir at /runs (ro),
# so a bake-off can load best.pt by that exact path. yolo-api sees the same
# files under this host root for existence checks.
RUNS_HOST_ROOT = Path(os.environ.get('LEGACY_TRAIN_RUNS_ROOT', '/data/curation_train_data/runs'))
# Roots scanned for frozen evaluation datasets (any dir with TEST_FROZEN.json).
# Adding a dataset = freeze a dir under one of these; no code change needed.
EVAL_DATASET_ROOTS: list[tuple[str, Path]] = [
    ('curated', Path('/data/curation_train_data/lpr_exports')),
    ('public', Path('/data/datasets/plates')),
    ('sample', Path('/data/datasets/plates/samples')),
]
# Baseline-model registry (public/commercial detectors). Add a model = one entry
# in this JSON; no code change. Lives next to the harness so the evaluator and
# the API share it.
BASELINES_PATH = Path(__file__).resolve().parents[3] / 'scripts/curation/bakeoff/baselines.json'
_JOB_ID_RE = re.compile(r'[A-Za-z0-9_.:-]{1,64}')


def _checkpoint_exists(checkpoint_path: str) -> bool:
    """True if a run's checkpoint is on disk (translating /runs -> host root)."""
    if checkpoint_path.startswith('/runs/'):
        candidate = RUNS_HOST_ROOT / checkpoint_path[len('/runs/') :]
    else:
        candidate = Path(checkpoint_path)
    return candidate.is_file()


def _infer_model_size(job_id: str, model_size: str | None) -> str | None:
    """Size from status.json, else parsed from a ``..._yolo26n`` job-id suffix."""
    if model_size:
        return model_size
    m = re.search(r'yolo\d+([nsmlx])$', job_id)
    return m.group(1) if m else None


def _safe_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=400, detail=f'invalid job_id: {job_id!r}')
    return job_id


class BakeoffModelSpec(BaseModel):
    """One model to score. Mirrors the harness CLI flags."""

    backend: str = Field(
        description='ultralytics | triton | open-image-models | lpdnet | two-stage'
    )
    name: str
    mode: str | None = None  # full | crop | both (run in source-frame and/or vehicle-crop mode)
    weights: str | None = None
    imgsz: int | None = None
    device: str | None = None
    plate_class_id: int | None = None
    lpdnet_variant: str | None = None
    triton_url: str | None = None
    triton_model: str | None = None
    vehicle_weights: str | None = None
    vehicle_classes: str | None = None
    vehicle_imgsz: int | None = None
    lpr_backend: str | None = None
    lpr_imgsz: int | None = None
    training_data: str | None = None


class DatasetRef(BaseModel):
    """One frozen evaluation dataset (a column in the matrix)."""

    path: str
    name: str | None = None


class BakeoffRequest(BaseModel):
    """Body for ``POST /bakeoff/run``.

    Provide ``datasets`` (the matrix form: every model is scored on every
    dataset) or the legacy single ``dataset``. ``models`` may use ``mode:
    'both'`` to score full-frame and vehicle-crop.
    """

    dataset: str | None = Field(default=None, description='Legacy single frozen export root')
    datasets: list[DatasetRef] | None = Field(default=None, description='Matrix: frozen datasets')
    models: list[BakeoffModelSpec] = Field(default_factory=list)
    verify_frozen: bool = True
    job_id: str | None = None
    # Optional: export a trained checkpoint to portable ONNX (fp32/fp16/int8)
    # first, then score those variants in this same job (seamless export ->
    # benchmark -> matrix -> frontend QuantizationPanel). See
    # bakeoff_runner._quantize_and_variant_models for the block shape.
    quantize: dict[str, Any] | None = Field(
        default=None, description='Auto-export + score quantized ONNX variants'
    )


@router.post('/bakeoff/run')
async def bakeoff_run(payload: BakeoffRequest) -> dict[str, Any]:
    """Enqueue a bake-off job for the evaluator container to execute."""
    if not payload.models and not payload.quantize:
        raise HTTPException(status_code=400, detail='no models specified')
    if not payload.datasets and not payload.dataset:
        raise HTTPException(status_code=400, detail='provide datasets or dataset')
    job_id = _safe_job_id(payload.job_id or datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    out_dir = OUT_DIR / job_id
    spec: dict[str, Any] = {
        'job_id': job_id,
        'verify_frozen': payload.verify_frozen,
        'out_dir': str(out_dir),
        'models': [m.model_dump(exclude_none=True) for m in payload.models],
    }
    if payload.quantize:
        spec['quantize'] = payload.quantize
    if payload.datasets:
        spec['datasets'] = [
            {'path': d.path, 'name': d.name or d.path.rstrip('/').split('/')[-1]}
            for d in payload.datasets
        ]
    else:
        spec['dataset'] = payload.dataset
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    (JOBS_DIR / f'{job_id}.job.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')

    # Auto-claim the configured GPU-resident containers exactly like training:
    # set the lock (closes the claim->scan race) and stop them now so the
    # evaluator starts on free GPUs. The arbiter reconcile loop keeps them
    # down while the job.json is in the queue and brings them back
    # automatically once it's done. On a generic install with no configured
    # containers this is a no-op (see GpuArbiterConfig / test_gpu_arbiter_config.py).
    try:
        from src.services.training.gpu_arbiter import set_training_lock, stop_gpu_services

        set_training_lock('0,1')
        action = await stop_gpu_services()
        logger.info('bakeoff_gpu_claimed', job_id=job_id, action=action.action)
    except Exception as exc:
        logger.warning('bakeoff_gpu_claim_failed', job_id=job_id, error=str(exc))

    logger.info(
        'bakeoff_enqueued',
        job_id=job_id,
        n_models=len(payload.models),
        n_datasets=len(payload.datasets) if payload.datasets else 1,
    )
    return {'status': 'enqueued', 'job_id': job_id, 'out_dir': str(out_dir)}


@router.get('/bakeoff/eval_datasets')
async def bakeoff_eval_datasets() -> dict[str, Any]:
    """Auto-discover frozen evaluation datasets (matrix columns).

    Any directory with a ``TEST_FROZEN.json`` under the curated/public/sample
    roots is selectable --- so adding a dataset is just freezing a dir, no code
    change. Returns name, path, kind, and the frozen label count.
    """
    seen: set[str] = set()
    datasets: list[dict[str, Any]] = []
    for kind, root in EVAL_DATASET_ROOTS:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            lock = d / 'TEST_FROZEN.json'
            if not d.is_dir() or not lock.is_file() or str(d) in seen:
                continue
            seen.add(str(d))
            n_test = None
            sha = None
            try:
                meta = json.loads(lock.read_text(encoding='utf-8'))
                n_test = meta.get('n_label_files')
                sha = meta.get('frozen_test_sha')
            except (OSError, ValueError):
                pass
            datasets.append(
                {'name': d.name, 'path': str(d), 'kind': kind, 'n_test': n_test, 'frozen_sha': sha}
            )
    return {'datasets': datasets, 'count': len(datasets)}


@router.get('/bakeoff/baseline_models')
async def bakeoff_baseline_models() -> dict[str, Any]:
    """Public/commercial baseline detectors from the editable registry."""
    try:
        reg = json.loads(BASELINES_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('bakeoff_baselines_read_failed', error=str(exc))
        return {'baselines': [], 'count': 0}
    baselines = [b for b in reg.get('baselines', []) if isinstance(b, dict)]
    return {'baselines': baselines, 'count': len(baselines)}


@router.get('/bakeoff/matrix/{job_id}')
async def bakeoff_matrix(job_id: str) -> dict[str, Any]:
    """Return the model x dataset matrix.json for a finished matrix job."""
    mf = OUT_DIR / _safe_job_id(job_id) / 'matrix.json'
    if not mf.is_file():
        raise HTTPException(status_code=404, detail=f'no matrix for job {job_id} (not done?)')
    return json.loads(mf.read_text(encoding='utf-8'))


@router.get('/bakeoff/trained_models')
async def bakeoff_trained_models(limit: int = 100) -> dict[str, Any]:
    """Finished training runs (with a checkpoint) selectable as bake-off contenders.

    Lets the UI pick an already-trained model directly from the backend
    instead of downloading + re-uploading weights. The evaluator container
    mounts ``/runs`` read-only, so it loads each run's ``best.pt`` by the
    same path the trainer recorded in ``status.json``. Future progressive
    sizes (s/m/l) appear here automatically as they finish.
    """
    from src.services.training.jobs import list_runs

    runs = await list_runs(limit=limit, offset=0)
    models: list[dict[str, Any]] = []
    for r in runs:
        if r.state != 'finished' or not r.checkpoint_path:
            continue
        if not _checkpoint_exists(r.checkpoint_path):
            continue
        models.append(
            {
                'run_id': r.job_id,
                'name': r.job_id,
                'model_size': _infer_model_size(r.job_id, getattr(r, 'model_size', None)),
                'checkpoint_path': r.checkpoint_path,
                'map50': (r.best_metric or {}).get('map50'),
                'finished_at': r.finished_at,
                'campaign_id': r.campaign_id,
            }
        )
    return {'models': models, 'count': len(models)}


@router.get('/bakeoff/runs')
async def bakeoff_runs() -> dict[str, Any]:
    """List bake-off runs (most recent first) with their state."""
    runs: list[dict[str, Any]] = []
    if OUT_DIR.is_dir():
        for d in OUT_DIR.iterdir():
            status_file = d / 'status.json'
            if not status_file.is_file():
                continue
            try:
                st = json.loads(status_file.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                continue
            runs.append(
                {
                    'job_id': d.name,
                    'state': st.get('state'),
                    'models': st.get('models', []),
                    'started_at': st.get('started_at'),
                    'finished_at': st.get('finished_at'),
                }
            )

    # Newest first by start time (fall back to the dir mtime), NOT by job-id
    # name --- letter-prefixed ids (sample_, roboflow_) would otherwise sort
    # above the timestamped/ours_ runs and bury the most recent ones.
    def _sort_key(r: dict[str, Any]) -> str:
        return r.get('started_at') or r.get('finished_at') or ''

    runs.sort(key=_sort_key, reverse=True)
    return {'runs': runs}


@router.get('/bakeoff/status/{job_id}')
async def bakeoff_status(job_id: str) -> dict[str, Any]:
    """Return the evaluator's status.json for a job."""
    sf = OUT_DIR / _safe_job_id(job_id) / 'status.json'
    if not sf.is_file():
        raise HTTPException(status_code=404, detail=f'no status for job {job_id}')
    return json.loads(sf.read_text(encoding='utf-8'))


@router.get('/bakeoff/results/{job_id}')
async def bakeoff_results(job_id: str) -> dict[str, Any]:
    """Return the ranked comparison rows for a finished job."""
    cf = OUT_DIR / _safe_job_id(job_id) / 'comparison.json'
    if not cf.is_file():
        raise HTTPException(status_code=404, detail=f'no comparison for job {job_id} (not done?)')
    return json.loads(cf.read_text(encoding='utf-8'))

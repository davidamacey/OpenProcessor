"""Bake-off API — trigger model-comparison runs and serve their results.

yolo-api pins an older Ultralytics release and cannot load every model
family the harness benchmarks, so it does NOT run the bake-off itself.
Instead it writes a ``<job_id>.job.json`` into a shared dir that an
on-demand evaluator container watches (same pattern as the training
router). The evaluator scores every model on the frozen test split, logs
to MLflow, and writes ``status.json`` + ``comparison.json`` back here for
the UI to read.

Endpoints:
    POST /bakeoff/run               enqueue a comparison job
    GET  /bakeoff/runs              list past/current runs
    GET  /bakeoff/status/{id}       job progress (running/done/error)
    GET  /bakeoff/results/{id}      ranked comparison rows for the UI
    GET  /bakeoff/matrix/{id}       model x dataset matrix
    GET  /bakeoff/eval_datasets     frozen evaluation datasets
    GET  /bakeoff/baseline_models   baseline-model registry (optionally per profile)
    GET  /bakeoff/trained_models    finished training runs usable as contenders
    GET  /bakeoff/profiles          available BakeoffProfiles

What a run measures (target class, cascade context classes, Triton model,
metric thresholds) comes from a ``BakeoffProfile``
(``scripts/curation/bakeoff/profile.py``) named by the request's optional
``profile`` field -- see ``docs/design/curation_design_rationale.md`` §8
("The detector bake-off harness"). The harness itself lives at
``scripts/curation/bakeoff/`` (not under ``src/`` -- see that package's
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

from src.config import get_curation_config, get_gpu_arbiter_config
from src.core.logging import get_logger
from src.routers.curation._common import router


logger = get_logger(__name__)


# Both yolo-api and the bake-off evaluator container mount the training-data
# root at the same absolute path, so these are shared between the two
# containers. `OP_BAKEOFF_*` env var names match the reference
# implementation's naming convention (post-rename, see the env-var-prefix
# unification commit). The *default* values (when the env var is unset) used
# to be owner-private absolute paths -- one of which named the location of a
# licensed proprietary image corpus and must never appear in this repo as a
# literal string (CFG-6). They are now derived from CurationConfig instead.
_curation_config = get_curation_config()

# JOBS_DIR defaults to GpuArbiterConfig.bakeoff_jobs_dir when configured, so
# gpu_arbiter.bakeoff_active() (the reconcile loop's "is a bake-off queued or
# running" check) watches the SAME directory this router writes job.json
# into, without requiring the operator to set the OP_BAKEOFF_JOBS_DIR env var
# and the GpuArbiterConfig field to the same value independently. Falls back
# to a state_dir-relative path (same precedent as OP_TRAIN_STAGING below)
# when neither is set.
_configured_jobs_dir = get_gpu_arbiter_config().bakeoff_jobs_dir
JOBS_DIR = Path(
    os.environ.get(
        'OP_BAKEOFF_JOBS_DIR',
        _configured_jobs_dir or str(_curation_config.state_dir / 'bakeoff_jobs'),
    )
)
OUT_DIR = Path(
    os.environ.get('OP_BAKEOFF_OUT_DIR', str(_curation_config.state_dir / 'bakeoff_out'))
)
# Training runs land here. The trainer records checkpoint_path as a /runs/...
# container path; the evaluator container mounts the same dir at /runs (ro),
# so a bake-off can load best.pt by that exact path. yolo-api sees the same
# files under this host root for existence checks.
RUNS_HOST_ROOT = Path(
    os.environ.get('OP_TRAIN_RUNS_ROOT', str(_curation_config.state_dir / 'training_runs'))
)
# Roots scanned for frozen evaluation datasets (any dir with TEST_FROZEN.json).
# Adding a dataset = freeze a dir under one of these; no code change needed.
# Rooted under CurationConfig.bakeoff_eval_root (env OP_BAKEOFF_EVAL_ROOT) --
# an operator with an existing frozen-dataset tree overrides that one var
# rather than three independent absolute-path defaults.
EVAL_DATASET_ROOTS: list[tuple[str, Path]] = [
    ('curated', _curation_config.bakeoff_eval_root / 'curated'),
    ('public', _curation_config.bakeoff_eval_root / 'public'),
    ('sample', _curation_config.bakeoff_eval_root / 'sample'),
]
# Baseline-model registry (public/commercial detectors). Add a model = one entry
# in this JSON; no code change. Lives next to the harness so the evaluator and
# the API share it. A profile may name its own registry (baselines_path); the
# OP_BAKEOFF_BASELINES_PATH env var replaces the default file.
_DEFAULT_BASELINES_PATH = (
    Path(__file__).resolve().parents[3] / 'scripts/curation/bakeoff/baselines.json'
)
BASELINES_PATH = Path(os.environ.get('OP_BAKEOFF_BASELINES_PATH') or _DEFAULT_BASELINES_PATH)
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


def _check_profile(spec: str | None) -> None:
    """400 on an unknown profile name.

    A ``.json`` path is passed through unchecked: it is resolved inside the
    evaluator container, whose filesystem may differ from this one's.
    """
    if not spec or spec.endswith('.json'):
        return
    from scripts.curation.bakeoff.profile import resolve_profile

    try:
        resolve_profile(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class BakeoffModelSpec(BaseModel):
    """One model to score. Mirrors the harness CLI flags."""

    backend: str = Field(
        description='ultralytics | triton | open-image-models | lpdnet | two-stage '
        '| onnxruntime | coreml'
    )
    name: str
    # Per-model BakeoffProfile override (else the request-level profile).
    profile: str | None = None
    mode: str | None = None  # full | crop | both (run in source-frame and/or crop mode)
    weights: str | None = None
    imgsz: int | None = None
    device: str | None = None
    pred_class_id: int | None = None
    gt_class_id: int | None = None
    gt_class_name: str | None = None
    lpdnet_variant: str | None = None
    triton_url: str | None = None
    triton_model: str | None = None
    # Coarse stage for --mode crop / two-stage. Unset fields come from the
    # profile (context_class_ids / context_weights / ...).
    primary_weights: str | None = None
    primary_classes: str | None = None
    primary_imgsz: int | None = None
    secondary_backend: str | None = None
    secondary_imgsz: int | None = None
    training_data: str | None = None


class DatasetRef(BaseModel):
    """One frozen evaluation dataset (a column in the matrix)."""

    path: str
    name: str | None = None


class BakeoffRequest(BaseModel):
    """Body for ``POST /bakeoff/run``.

    Provide ``datasets`` (the matrix form: every model is scored on every
    dataset) or the legacy single ``dataset``. ``models`` may use ``mode:
    'both'`` to score full-frame and crop mode.
    """

    dataset: str | None = Field(default=None, description='Legacy single frozen export root')
    profile: str | None = Field(
        default=None,
        description='BakeoffProfile name (see GET /bakeoff/profiles) or a profile .json path '
        'on the evaluator; omitted = the evaluator default (generic)',
    )
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
    if payload.quantize and payload.quantize.get('coreml'):
        from scripts.curation.bakeoff.bakeoff_runner import COREML_UNAVAILABLE

        raise HTTPException(status_code=400, detail=COREML_UNAVAILABLE)
    if not payload.datasets and not payload.dataset:
        raise HTTPException(status_code=400, detail='provide datasets or dataset')
    _check_profile(payload.profile)
    for m in payload.models:
        _check_profile(m.profile)
    job_id = _safe_job_id(payload.job_id or datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    out_dir = OUT_DIR / job_id
    spec: dict[str, Any] = {
        'job_id': job_id,
        'verify_frozen': payload.verify_frozen,
        'out_dir': str(out_dir),
        'models': [m.model_dump(exclude_none=True) for m in payload.models],
    }
    if payload.profile:
        spec['profile'] = payload.profile
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
                # W1 renamed the lock key to test_label_sha; fall back to the
                # legacy key for locks written before that (see freeze.py).
                sha = meta.get('test_label_sha', meta.get('frozen_test_sha'))
            except (OSError, ValueError):
                pass
            datasets.append(
                {'name': d.name, 'path': str(d), 'kind': kind, 'n_test': n_test, 'frozen_sha': sha}
            )
    return {'datasets': datasets, 'count': len(datasets)}


@router.get('/bakeoff/profiles')
async def bakeoff_profiles() -> dict[str, Any]:
    """Registered + example BakeoffProfiles, flagging the default.

    ``kind`` says where a profile comes from (``registered`` in code,
    ``example`` from ``examples/<name>/profile.json``, ``configured`` for a
    default that is neither -- e.g. ``OP_BAKEOFF_PROFILE`` naming a .json
    path). ``default`` marks the profile a job gets when it names none
    (``OP_BAKEOFF_PROFILE``, else ``generic`` plus ``OP_BAKEOFF_PROFILE_*``
    overrides); the default row carries those effective field values. The
    top-level ``default_profile`` repeats its name (null if the configured
    default does not resolve -- see ``default_error``). Resolved from this
    API process's environment, which the deployment is expected to share
    with the evaluator.
    """
    from scripts.curation.bakeoff.profile import registered_profiles, resolve_profile

    # Example profiles are opt-in (loaded by path), so they are no longer
    # listed; the typed rewrite of this route lands with the W4 API wave.
    profiles: list[dict[str, Any]] = [
        {**prof.to_dict(), 'kind': 'registered', 'default': False}
        for prof in registered_profiles().values()
    ]

    body: dict[str, Any] = {'default_profile': None}
    try:
        default = resolve_profile(None)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning('bakeoff_default_profile_invalid', error=str(exc))
        body['default_error'] = str(exc)
    else:
        body['default_profile'] = default.name
        row = next((p for p in profiles if p['name'] == default.name), None)
        if row is None:
            profiles.append({**default.to_dict(), 'kind': 'configured', 'default': True})
        else:
            row.update(default.to_dict(), default=True)
    return {'profiles': profiles, 'count': len(profiles), **body}


@router.get('/bakeoff/baseline_models')
async def bakeoff_baseline_models(profile: str | None = None) -> dict[str, Any]:
    """Baseline detectors from the editable registry.

    With ``?profile=<name>`` (a registered/example name, not a path) the
    profile's own ``baselines_path`` is read instead, falling back to the
    default registry when it names none.
    """
    path = BASELINES_PATH
    if profile:
        if '/' in profile or profile.endswith('.json'):
            raise HTTPException(status_code=400, detail='profile must be a profile name')
        from scripts.curation.bakeoff.profile import resolve_baselines_path, resolve_profile

        try:
            path = resolve_baselines_path(resolve_profile(profile), BASELINES_PATH)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        reg = json.loads(path.read_text(encoding='utf-8'))
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
        run_eval = r.eval or {}
        models.append(
            {
                'run_id': r.job_id,
                'name': r.job_id,
                'model_size': _infer_model_size(r.job_id, getattr(r, 'model_size', None)),
                'checkpoint_path': r.checkpoint_path,
                'map50': run_eval.get('map50'),
                'map50_split': run_eval.get('split'),
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

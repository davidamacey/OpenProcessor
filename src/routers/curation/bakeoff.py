"""Model comparison (bake-off) API: enqueue comparison jobs and serve their results.

yolo-api pins an older Ultralytics release and cannot load every model
family the harness scores, so it does NOT run the comparison itself. It
resolves the request (eval datasets, training runs, class mappings; see
``src/services/curation/bakeoff_jobs.py``) into a ``<job_id>.job.json`` in
``JOBS_DIR``, which the evaluator container watches. The evaluator scores
every model on every dataset's frozen test split and writes ``status.json``,
``<dataset dir>/comparison.json`` and ``matrix.json`` under
``OUT_DIR/<job_id>/`` for these routes to serve.

Endpoints:
    GET  /bakeoff/eval_datasets     exports + external frozen sets usable as eval data
    GET  /bakeoff/trained_models    finished training runs usable as contenders
    GET  /bakeoff/profiles          registered profiles + the configured default
    GET  /bakeoff/baseline_models   external-model registry (optionally per profile)
    POST /bakeoff/run               enqueue a comparison job
    GET  /bakeoff/status/{id}       job progress (queued/running/done/error)
    GET  /bakeoff/runs              past/current jobs, newest first
    GET  /bakeoff/results/{id}      ranked comparison for one dataset of a job
    GET  /bakeoff/matrix/{id}       model x dataset matrix

GPU: enqueueing stops the configured GPU-resident containers
(``GpuArbiterConfig``) and answers 409 if it cannot. The arbiter's reconcile
loop keeps them stopped while any job file is queued in ``JOBS_DIR`` and
restarts them once the evaluator moves it to ``done/``. A job queued with no
evaluator running keeps them stopped; its status stays ``queued`` and
deleting the job file releases the GPU on the next reconcile tick.

What a run measures (classes, thresholds, rank metric, cascade context
classes) comes from a ``BakeoffProfile`` (``scripts/curation/bakeoff/profile.py``)
-- see ``docs/design/curation_design_rationale.md`` §8 ("The detector
bake-off harness").
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from src.config import get_curation_config, get_gpu_arbiter_config
from src.core.logging import get_logger
from src.routers.curation._bakeoff_models import (
    BakeoffComparison,
    BakeoffMatrix,
    BakeoffProfileList,
    BakeoffProfileRow,
    BakeoffRunAccepted,
    BakeoffRunList,
    BakeoffRunRequest,
    BakeoffRunRow,
    BakeoffStatus,
    BaselineModelList,
    EvalDataset,
    EvalDatasetList,
    JobProgress,
    TrainedModelList,
)
from src.routers.curation._common import router
from src.services.curation import bakeoff_jobs, eval_datasets


logger = get_logger(__name__)

# The arbiter's reconcile loop watches this same dir (bakeoff_active()).
JOBS_DIR = Path(get_gpu_arbiter_config().bakeoff_jobs_dir)
OUT_DIR = Path(
    os.environ.get('OP_BAKEOFF_OUT_DIR', str(get_curation_config().state_dir / 'bakeoff_out'))
)


def _raise(exc: bakeoff_jobs.BakeoffRequestError) -> NoReturn:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _read_result[M: BaseModel](path: Path, model: type[M], missing: str) -> M:
    """Validate an evaluator-written file; 404 if absent, 409 if not schema v2."""
    if not path.is_file():
        raise HTTPException(status_code=404, detail=missing)
    try:
        return model.model_validate(json.loads(path.read_text(encoding='utf-8')))
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f'bake-off result {path.name} has an unsupported schema (schema_version != 2)',
        ) from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(path)


@router.get('/bakeoff/eval_datasets', response_model=EvalDatasetList)
async def bakeoff_eval_datasets(
    source: Literal['export', 'external'] | None = None,
) -> EvalDatasetList:
    """Eval datasets: every export with a labelled test split, plus frozen external sets.

    Class counts and test-split hashes are computed from the files; see
    ``src/services/curation/eval_datasets.py``.
    """
    rows = [EvalDataset(**r.to_wire()) for r in eval_datasets.list_eval_datasets(source)]
    return EvalDatasetList(datasets=rows, count=len(rows))


@router.get('/bakeoff/trained_models', response_model=TrainedModelList)
async def bakeoff_trained_models(
    dataset_id: str | None = None, limit: int = 100
) -> TrainedModelList:
    """Finished training runs (checkpoint on disk) selectable as contenders.

    With ``?dataset_id=`` each row carries ``for_dataset``: same export /
    same frozen test split as the dataset, how many of its classes map onto
    the dataset's, and the train/test image overlap (a leakage warning).
    """
    try:
        models = await bakeoff_jobs.list_trained_models(dataset_id, limit)
    except bakeoff_jobs.BakeoffRequestError as exc:
        _raise(exc)
    return TrainedModelList(models=models, count=len(models))


def _profile_row(
    prof: Any, kind: Literal['registered', 'configured'], *, default: bool
) -> BakeoffProfileRow:
    d = prof.to_dict()
    return BakeoffProfileRow(
        **{k: d[k] for k in BakeoffProfileRow.model_fields if k in d},
        kind=kind,
        default=default,
    )


@router.get('/bakeoff/profiles', response_model=BakeoffProfileList)
async def bakeoff_profiles() -> BakeoffProfileList:
    """Registered profiles, flagging the default a job gets when it names none.

    ``kind``: ``registered`` (in code) or ``configured`` (the default,
    resolved from ``OP_BAKEOFF_PROFILE`` naming a ``.json`` path). Example
    profiles under ``examples/`` are opt-in by path and not listed. The
    default row carries its effective values (``OP_BAKEOFF_PROFILE_*``
    overrides included). Resolved from this process's environment, which the
    deployment shares with the evaluator.
    """
    from scripts.curation.bakeoff.profile import registered_profiles, resolve_profile

    rows = [_profile_row(p, 'registered', default=False) for p in registered_profiles().values()]
    try:
        default = resolve_profile(None)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning('bakeoff_default_profile_invalid', error=str(exc))
        return BakeoffProfileList(
            profiles=rows, count=len(rows), default_profile=None, default_error=str(exc)
        )
    idx = next((i for i, r in enumerate(rows) if r.name == default.name), None)
    if idx is None:
        rows.append(_profile_row(default, 'configured', default=True))
    else:
        rows[idx] = _profile_row(default, 'registered', default=True)
    return BakeoffProfileList(profiles=rows, count=len(rows), default_profile=default.name)


@router.get('/bakeoff/baseline_models', response_model=BaselineModelList)
async def bakeoff_baseline_models(profile: str | None = None) -> BaselineModelList:
    """External models from the baseline registry (empty by default).

    With ``?profile=<name>`` (a registered name, not a path) the profile's own
    ``baselines_path`` is read instead.
    """
    try:
        path = bakeoff_jobs.baselines_path_for(profile)
    except bakeoff_jobs.BakeoffRequestError as exc:
        _raise(exc)
    baselines = bakeoff_jobs.load_baselines(path)
    return BaselineModelList(baselines=baselines, count=len(baselines))


@router.post('/bakeoff/run', response_model=BakeoffRunAccepted)
async def bakeoff_run(payload: BakeoffRunRequest) -> BakeoffRunAccepted:
    """Enqueue a comparison job for the evaluator container.

    Order: resolve + validate (400/422) -> ``status.json`` ``queued`` -> job
    file (atomic) -> stop GPU-resident containers. If they cannot be stopped
    the job file is removed, the status set to ``error`` and the answer is 409.
    """
    from src.services.training import gpu_arbiter

    try:
        spec, accepted = await bakeoff_jobs.build_job_spec(payload, out_root=OUT_DIR)
    except bakeoff_jobs.BakeoffRequestError as exc:
        _raise(exc)
    job_file = JOBS_DIR / f'{spec.job_id}.job.json'
    status_file = OUT_DIR / spec.job_id / 'status.json'
    if job_file.exists() or status_file.exists():
        raise HTTPException(status_code=409, detail=f'bake-off job {spec.job_id} already exists')

    models = [m.model for m in spec.models]
    status = BakeoffStatus(
        schema_version=2,
        job_id=spec.job_id,
        state='queued',
        profile=accepted.profile,
        datasets=[d.id for d in spec.datasets],
        models=models,
        started_at=None,
        finished_at=None,
        progress=JobProgress(done=0, total=len(spec.datasets) * len(models)),
        completed=[],
        failed=[],
        error=None,
    )
    _write_json(status_file, status.model_dump(mode='json'))
    _write_json(job_file, spec.model_dump(mode='json'))
    try:
        action = await gpu_arbiter.stop_gpu_services()
    except gpu_arbiter.GpuArbiterStopFailedError as exc:
        job_file.unlink(missing_ok=True)
        status.state, status.error = 'error', f'could not free the GPU: {exc}'
        status.finished_at = datetime.now(UTC).isoformat()
        _write_json(status_file, status.model_dump(mode='json'))
        logger.warning('bakeoff_gpu_claim_failed', job_id=spec.job_id, error=str(exc))
        raise HTTPException(status_code=409, detail=status.error) from exc
    logger.info(
        'bakeoff_enqueued',
        job_id=spec.job_id,
        gpu_action=action.action,
        n_models=len(models),
        n_datasets=len(spec.datasets),
    )
    return accepted


@router.get('/bakeoff/status/{job_id}', response_model=BakeoffStatus)
async def bakeoff_status(job_id: str) -> BakeoffStatus:
    """The job's ``status.json`` (``queued`` until the evaluator picks it up)."""
    job_id = _safe(job_id)
    return _read_result(
        OUT_DIR / job_id / 'status.json', BakeoffStatus, f'no status for job {job_id}'
    )


def _safe(job_id: str) -> str:
    try:
        return bakeoff_jobs.safe_job_id(job_id)
    except bakeoff_jobs.BakeoffRequestError as exc:
        _raise(exc)


@router.get('/bakeoff/runs', response_model=BakeoffRunList)
async def bakeoff_runs() -> BakeoffRunList:
    """Comparison jobs, newest first. Directories without a v2 ``status.json`` are skipped."""
    rows: list[tuple[str, BakeoffRunRow]] = []
    if OUT_DIR.is_dir():
        for d in OUT_DIR.iterdir():
            status_file = d / 'status.json'
            if not status_file.is_file():
                continue
            try:
                st = BakeoffStatus.model_validate(json.loads(status_file.read_text('utf-8')))
            except (ValueError, ValidationError):
                continue
            row = BakeoffRunRow(**st.model_dump(include=set(BakeoffRunRow.model_fields)))
            # Queued jobs have no started_at yet: fall back to the status mtime.
            mtime = datetime.fromtimestamp(status_file.stat().st_mtime, UTC).isoformat()
            rows.append((st.started_at or st.finished_at or mtime, row))
    rows.sort(key=lambda kr: kr[0], reverse=True)
    return BakeoffRunList(runs=[r for _, r in rows])


@router.get('/bakeoff/results/{job_id}', response_model=BakeoffComparison)
async def bakeoff_results(job_id: str, dataset_id: str | None = None) -> BakeoffComparison:
    """Ranked comparison of every model on one dataset (default: the job's first)."""
    job_id = _safe(job_id)
    if dataset_id is None:
        status = _read_result(
            OUT_DIR / job_id / 'status.json', BakeoffStatus, f'no comparison for job {job_id}'
        )
        if not status.datasets:
            raise HTTPException(status_code=404, detail=f'job {job_id} has no datasets')
        dataset_id = status.datasets[0]
    if not eval_datasets.DATASET_ID_RE.fullmatch(dataset_id):
        raise HTTPException(status_code=400, detail=f'invalid dataset id {dataset_id!r}')
    dir_name = dataset_id.replace(':', '__').replace('/', '__')
    return _read_result(
        OUT_DIR / job_id / dir_name / 'comparison.json',
        BakeoffComparison,
        f'no comparison for job {job_id} on {dataset_id} (not done?)',
    )


@router.get('/bakeoff/matrix/{job_id}', response_model=BakeoffMatrix)
async def bakeoff_matrix(job_id: str) -> BakeoffMatrix:
    """Model x dataset matrix of a finished job; ``best`` lists every tied winner."""
    job_id = _safe(job_id)
    return _read_result(
        OUT_DIR / job_id / 'matrix.json', BakeoffMatrix, f'no matrix for job {job_id} (not done?)'
    )

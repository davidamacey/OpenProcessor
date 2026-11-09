"""Model comparison (bake-off) API: enqueue comparison jobs and serve their results.

api pins an older Ultralytics release and cannot load every model
family the harness scores, so it does NOT run the comparison itself. It
resolves the request (eval datasets, training runs, class mappings; see
``src/services/curation/bakeoff_jobs.py`` and its siblings) into the parameters of a
``bakeoff`` job on the ``gpu_eval`` queue (``start_job``, with a ``bakeoff_runs`` row), which
the evaluator worker runs (the ``bakeoff_tasks`` module of the bakeoff package). The evaluator scores
every model on every dataset's frozen test split and writes ``status.json``,
``<dataset dir>/comparison.json`` and ``matrix.json`` under
``_out_dir()/<job_id>/`` for these routes to serve.

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

GPU: enqueueing stops the GPU-resident containers configured in
``GpuArbiterConfig`` and answers 409 (cancelling the job) if it cannot. When
``OP_BAKEOFF_HOST_GPUS`` is set (the host GPU ids the evaluator container is
attached to), only containers scoped (``GpuArbiterConfig.container_gpus``) to
those ids are stopped. Unset stops everything configured. The arbiter's reconcile
loop keeps the scoped set stopped while a ``bakeoff`` job is active and restarts them
once it ends.

What a run measures (classes, thresholds, rank metric, cascade context
classes) comes from a ``BakeoffProfile`` (``scripts/curation/bakeoff/profile.py``)
-- see ``docs/design/curation_design_rationale.md`` §8 ("The detector
bake-off harness").
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from src.config import get_curation_config, get_gpu_arbiter_config
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.core.logging import get_logger
from src.routers.curation._common import router
from src.schemas.bakeoff.datasets import EvalDataset, EvalDatasetList
from src.schemas.bakeoff.models import (
    BakeoffProfileList,
    BakeoffProfileRow,
    BaselineModelList,
    TrainedModelList,
)
from src.schemas.bakeoff.results import BakeoffComparison, BakeoffMatrix
from src.schemas.bakeoff.run import (
    BakeoffRunAccepted,
    BakeoffRunList,
    BakeoffRunRequest,
    BakeoffRunRow,
    BakeoffStatus,
    JobProgress,
)
from src.services.classes.registry import ClassRegistryError, get_class_registry
from src.services.curation import bakeoff_jobs, bakeoff_resolve, eval_datasets
from src.services.jobs.service import cancel_job
from src.services.training.run_retention import bakeoff_out_dir
from src.utils.atomic_write import atomic_write_json
from src.utils.upstream_errors import UpstreamError


logger = get_logger(__name__)


# Project-scoped (the project-isolation design §5.3): each project's bake-off results live under
# its own ``<CurationConfig.bakeoff_jobs_dir>/out`` (a PROJECT_SCOPED_FIELDS entry), so no
# project's results leak into another's. Resolved at call time (never a module-level constant)
# so a later bind_project is always picked up. The directory name is what lane 5E renames: it
# holds results only now; no job file is written there.
def _out_dir() -> Path:
    return bakeoff_out_dir(Path(get_curation_config().bakeoff_jobs_dir))


def _raise(exc: bakeoff_resolve.BakeoffRequestError) -> NoReturn:
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
    atomic_write_json(path, payload, default=None, indent=2)


@router.get('/bakeoff/eval_datasets', response_model=EvalDatasetList)
async def bakeoff_eval_datasets(
    source: Literal['export', 'external'] | None = None,
) -> EvalDatasetList:
    """Eval datasets: every export with a labelled test split, plus frozen external sets.

    Class counts and test-split hashes are computed from the files; see
    ``src/services/curation/eval_datasets.py``.
    """
    names = await _registry_names_by_id()
    rows = [EvalDataset(**r.to_wire(names)) for r in eval_datasets.list_eval_datasets(source)]
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
    except bakeoff_resolve.BakeoffRequestError as exc:
        _raise(exc)
    return TrainedModelList(models=models, count=len(models))


async def _registry_names_by_id() -> dict[int, str]:
    """Live registry id -> name. A database failure is a 503, never an empty map."""
    try:
        snapshot = await get_class_registry().snapshot()
    except (ClassRegistryError, SQLAlchemyError) as exc:
        logger.warning('bakeoff_registry_unavailable', error=str(exc))
        raise UpstreamError('postgres', 'registry_unavailable') from exc
    return {c.class_id: c.class_name for c in snapshot.classes}


def _profile_row(
    prof: Any,
    kind: Literal['registered', 'configured'],
    *,
    default: bool,
    names_by_id: dict[int, str],
) -> BakeoffProfileRow:
    d = prof.to_dict()
    context_class_ids = d.get('context_class_ids', [])
    context_class_names = [names_by_id.get(cid, str(cid)) for cid in context_class_ids]
    return BakeoffProfileRow(
        **{k: d[k] for k in BakeoffProfileRow.model_fields if k in d},
        kind=kind,
        default=default,
        context_class_names=context_class_names,
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

    names_by_id = await _registry_names_by_id()
    rows = [
        _profile_row(p, 'registered', default=False, names_by_id=names_by_id)
        for p in registered_profiles().values()
    ]
    try:
        default = resolve_profile(None)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning('bakeoff_default_profile_invalid', error=str(exc))
        return BakeoffProfileList(
            profiles=rows, count=len(rows), default_profile=None, default_error=str(exc)
        )
    idx = next((i for i, r in enumerate(rows) if r.name == default.name), None)
    if idx is None:
        rows.append(_profile_row(default, 'configured', default=True, names_by_id=names_by_id))
    else:
        rows[idx] = _profile_row(default, 'registered', default=True, names_by_id=names_by_id)
    return BakeoffProfileList(profiles=rows, count=len(rows), default_profile=default.name)


@router.get('/bakeoff/baseline_models', response_model=BaselineModelList)
async def bakeoff_baseline_models(profile: str | None = None) -> BaselineModelList:
    """External models from the baseline registry (empty by default).

    With ``?profile=<name>`` (a registered name, not a path) the profile's own
    ``baselines_path`` is read instead.
    """
    try:
        path = bakeoff_resolve.baselines_path_for(profile)
    except bakeoff_resolve.BakeoffRequestError as exc:
        _raise(exc)
    baselines = bakeoff_resolve.load_baselines(path)
    return BaselineModelList(baselines=baselines, count=len(baselines))


@router.post('/bakeoff/run', response_model=BakeoffRunAccepted)
async def bakeoff_run(payload: BakeoffRunRequest, session: DbSessionDep) -> BakeoffRunAccepted:
    """Start a comparison as a ``bakeoff`` job for the evaluator worker.

    Order: resolve + validate (400/422) -> ``status.json`` ``queued`` -> ``start_job`` (the job
    row, its ``bakeoff_runs`` row and the queue message) -> stop GPU-resident containers. If they
    cannot be stopped the job is cancelled, the status set to ``error`` and the answer is 409.
    """
    from src.services.training.gpu_arbiter_docker import stop_gpu_services
    from src.services.training.gpu_arbiter_policy import (
        GpuArbiterStopFailedError,
        containers_to_stop,
    )

    try:
        spec, accepted = await bakeoff_jobs.build_job_spec(payload, out_root=_out_dir())
    except bakeoff_resolve.BakeoffRequestError as exc:
        _raise(exc)
    status_file = _out_dir() / spec.job_id / 'status.json'
    if status_file.exists():
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
    project = current_project_id()

    _write_json(status_file, status.model_dump(mode='json'))
    try:
        started = await bakeoff_jobs.start_bakeoff(session, project, spec)
    except Exception:
        status_file.unlink(missing_ok=True)
        raise
    if not started.created:
        raise HTTPException(status_code=409, detail=f'bake-off job {spec.job_id} already exists')
    scope = get_gpu_arbiter_config().bakeoff_host_gpus
    try:
        action = await stop_gpu_services(containers=containers_to_stop(scope) if scope else None)
    except GpuArbiterStopFailedError as exc:
        await cancel_job(session, started.job.id, project, requested_by='api')
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
        _out_dir() / job_id / 'status.json', BakeoffStatus, f'no status for job {job_id}'
    )


def _safe(job_id: str) -> str:
    try:
        return bakeoff_resolve.safe_job_id(job_id)
    except bakeoff_resolve.BakeoffRequestError as exc:
        _raise(exc)


@router.get('/bakeoff/runs', response_model=BakeoffRunList)
async def bakeoff_runs() -> BakeoffRunList:
    """Comparison jobs, newest first. Directories without a v2 ``status.json`` are skipped."""
    rows: list[tuple[str, BakeoffRunRow]] = []
    if _out_dir().is_dir():
        for d in _out_dir().iterdir():
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
            _out_dir() / job_id / 'status.json', BakeoffStatus, f'no comparison for job {job_id}'
        )
        if not status.datasets:
            raise HTTPException(status_code=404, detail=f'job {job_id} has no datasets')
        dataset_id = status.datasets[0]
    if not eval_datasets.DATASET_ID_RE.fullmatch(dataset_id):
        raise HTTPException(status_code=400, detail=f'invalid dataset id {dataset_id!r}')
    dir_name = dataset_id.replace(':', '__').replace('/', '__')
    return _read_result(
        _out_dir() / job_id / dir_name / 'comparison.json',
        BakeoffComparison,
        f'no comparison for job {job_id} on {dataset_id} (not done?)',
    )


@router.get('/bakeoff/matrix/{job_id}', response_model=BakeoffMatrix)
async def bakeoff_matrix(job_id: str) -> BakeoffMatrix:
    """Model x dataset matrix of a finished job; ``best`` lists every tied winner."""
    job_id = _safe(job_id)
    return _read_result(
        _out_dir() / job_id / 'matrix.json',
        BakeoffMatrix,
        f'no matrix for job {job_id} (not done?)',
    )

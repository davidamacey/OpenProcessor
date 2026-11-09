"""ST-4: keep-last retention for finished training runs.

A run is a ``training_runs`` row plus its ``jobs`` row; this module plans (and applies) a keep-last
prune of them that never touches:

* a **non-terminal** run (``queued``/``running``/``paused``/``cancelling``, and an ``interrupted``
  one awaiting resume);
* a run promoted to Triton (the caller passes the ``run_job_id`` of the project's live
  promotions, read from the registry in PostgreSQL);
* a run referenced by a queued/running bake-off job (``job_id`` appears as
  a ``models[].run_id`` (or the ``quantize.run_id``) in the params of an active ``bakeoff`` job).

Runs are ordered newest first by their job's creation time. Applying a prune removes each run's
artifact blobs under its ``artifact_prefix`` and its MLflow run before its rows.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

import httpx

from src.config.env import env_str
from src.config.project_context import current_project_id
from src.core.logging import get_logger
from src.db.repos import training_runs as runs_repo
from src.db.session import sync_project_tx, sync_session_factory
from src.domain import job_states as js
from src.services.training.bakeoff_pins import active_bakeoff_specs
from src.services.training.jobs import run_store


if TYPE_CHECKING:
    import uuid
    from pathlib import Path


logger = get_logger(__name__)

_MLFLOW_TIMEOUT_S = 10.0


def bakeoff_out_dir(bakeoff_results_dir: Path) -> Path:
    """A project's bake-off result dir: ``<bakeoff_results_dir>/out``. The one
    definition the router (serves results) and the pruner (deletes them)
    share, so each only ever touches its own project's results."""
    return bakeoff_results_dir / 'out'


def list_runs() -> list[tuple[uuid.UUID, str]]:
    """The bound project's runs as ``(job_id, status)``, newest first."""
    with (
        sync_session_factory()() as session,
        sync_project_tx(session, current_project_id()),
    ):
        return runs_repo.refs(session)


def collect_active_bakeoff_run_ids() -> set[str]:
    """Every training ``run_id`` a queued/running bake-off job still scores.

    Read from the job params in PostgreSQL; a database error propagates so a prune never
    proceeds without knowing which runs are pinned.
    """
    ids: set[str] = set()
    for spec in active_bakeoff_specs():
        for model in spec.get('models') or []:
            run_id = model.get('run_id') if isinstance(model, dict) else None
            if run_id:
                ids.add(str(run_id))
        quantize = spec.get('quantize')
        if isinstance(quantize, dict) and quantize.get('run_id'):
            ids.add(str(quantize['run_id']))
    return ids


def plan_run_prune(
    runs: list[tuple[uuid.UUID, str]],
    *,
    promoted_job_ids: set[str],
    bakeoff_run_ids: set[str],
    keep_last: int,
) -> list[uuid.UUID]:
    """Job ids to remove: terminal runs older than the newest ``keep_last`` of ``runs``
    (newest first, every state), skipping anything promoted or bake-off-active.

    ``keep_last <= 0`` keeps everything (returns an empty plan).
    """
    if keep_last <= 0:
        return []
    pinned = promoted_job_ids | bakeoff_run_ids
    return [
        job_id
        for job_id, status in runs[keep_last:]
        if status in js.TERMINAL and str(job_id) not in pinned
    ]


def _delete_mlflow_run(run_id: str) -> None:
    """Soft-delete one MLflow run over the tracking server's REST API (``runs/delete``, the call
    ``MlflowClient.delete_run`` makes; the API image carries no ``mlflow`` package). A run the
    server no longer has counts as deleted; any other failure raises."""
    base = env_str('MLFLOW_TRACKING_URI').rstrip('/')
    if not base:
        raise RuntimeError('MLFLOW_TRACKING_URI is not set')
    response = httpx.post(
        f'{base}/api/2.0/mlflow/runs/delete', json={'run_id': run_id}, timeout=_MLFLOW_TIMEOUT_S
    )
    if response.status_code == 404 or (
        response.status_code == 400 and 'RESOURCE_DOES_NOT_EXIST' in response.text
    ):
        return
    response.raise_for_status()


def apply_run_prune(job_ids: list[uuid.UUID]) -> dict[str, Any]:
    """Delete each planned run: its blobs, its MLflow run, then its rows.

    Fail closed: a blob or MLflow error keeps that run's rows (a later prune retries) and is
    listed under ``failed``; the other runs still go. Deleting an absent prefix or run is a no-op.
    """
    with (
        sync_session_factory()() as session,
        sync_project_tx(session, current_project_id()),
    ):
        rows = [r for r in (runs_repo.get(session, job_id) for job_id in job_ids) if r is not None]
    store = run_store()
    deletable: list[uuid.UUID] = []
    failed: list[dict[str, str]] = []
    for row in rows:
        try:
            store.delete_prefix(row.artifact_prefix)
            if row.mlflow_run_id:
                _delete_mlflow_run(row.mlflow_run_id)
        except (OSError, httpx.HTTPError, RuntimeError) as exc:
            logger.warning('run_prune_failed', job_id=str(row.job_id), error=str(exc))
            failed.append({'job_id': str(row.job_id), 'error': f'{type(exc).__name__}: {exc}'})
            continue
        deletable.append(row.job_id)
    with (
        sync_session_factory()() as session,
        sync_project_tx(session, current_project_id()),
    ):
        removed = runs_repo.delete_runs(session, deletable)
    for job_id in deletable:
        logger.info('run_pruned', job_id=str(job_id))
    return {'removed_runs': removed, 'failed': failed}


def plan_bakeoff_out_prune(out_dir: Path, keep_last: int) -> list[Path]:
    """Bake-off output dirs beyond ``keep_last``, newest-name-first.

    No pin logic: bake-off output is scored comparison output, not a
    training artifact anything else references by path.
    """
    if keep_last <= 0 or not out_dir.is_dir():
        return []
    candidates = sorted(
        (d for d in out_dir.iterdir() if d.is_dir()), key=lambda p: p.name, reverse=True
    )
    return candidates[keep_last:]


def apply_bakeoff_out_prune(paths: list[Path]) -> dict[str, int]:
    removed = 0
    for path in paths:
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.warning('bakeoff_out_prune_failed', path=str(path), error=str(exc))
            continue
        removed += 1
        logger.info('bakeoff_out_pruned', path=str(path))
    return {'removed': removed}


__all__ = [
    'apply_bakeoff_out_prune',
    'apply_run_prune',
    'collect_active_bakeoff_run_ids',
    'list_runs',
    'plan_bakeoff_out_prune',
    'plan_run_prune',
]

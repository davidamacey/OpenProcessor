"""Reading a run's manifest, artifacts and spec, and stamping a promotion onto the manifest.

Split out of ``jobs.py``. The manifest is ``training_runs.manifest`` (the trainer writes it when
the run ends); the metric artifacts are blobs under the run's ``artifact_prefix``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project_id
from src.core.logging import get_logger
from src.db.repos import training_runs as runs_repo
from src.db.session import async_session_factory, project_tx
from src.domain.training_keys import RUN_ARTIFACT_WHITELIST, run_object_key
from src.services.training.job_wire import _public_mlflow_url, _rewrite_eval_for_wire
from src.services.training.jobs import parse_run_id, run_store


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


async def _run(job_id: str) -> runs_repo.RunRow | None:
    run_id = parse_run_id(job_id)
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        return await session.run_sync(lambda sync: runs_repo.get(sync, run_id))


async def read_manifest(job_id: str) -> dict[str, Any] | None:
    """The run manifest or ``None`` while the run has not written one.

    The served ``results.eval`` / ``results.mlflow_run_url`` go through the same wire rewrite as
    ``TrainJobStatus`` (see ``_rewrite_eval_for_wire``/``_public_mlflow_url``): the manifest is a
    raw dict, not that model, so ``_prepare_status_for_wire`` does not cover it.
    """
    run = await _run(job_id)
    manifest = None if run is None else run.manifest
    if manifest is None:
        return None
    results = manifest.get('results')
    if isinstance(results, dict):
        results = dict(results)
        results['eval'] = _rewrite_eval_for_wire(results.get('eval'), job_id)
        results['mlflow_run_url'] = _public_mlflow_url(
            results.get('mlflow_run_id'), results.get('mlflow_experiment_id')
        )
        manifest = {**manifest, 'results': results}
    return manifest


async def read_artifact(job_id: str, name: str) -> Path | None:
    """The local path of a whitelisted run artifact, or ``None``.

    ``None`` covers every "don't serve this" case alike (name not in
    :data:`RUN_ARTIFACT_WHITELIST`, unknown run, missing file); the router turns any of them into
    a plain 404 so a probe can learn no more than "nothing there".
    """
    if name not in RUN_ARTIFACT_WHITELIST:
        return None
    run = await _run(job_id)
    if run is None:
        return None
    key = run_object_key(run.artifact_prefix, name)
    store = run_store()

    def _do() -> Path | None:
        return store.local_path(key) if store.exists(key) else None

    return await asyncio.to_thread(_do)


async def read_job_spec(job_id: str) -> dict[str, Any] | None:
    """The spec the run was started with (``training_runs.spec``), or ``None`` for an unknown run.

    Always present once the run is queued, unlike the manifest; promote uses it to find the
    registry snapshot pinned at submit time.
    """
    run = await _run(job_id)
    return None if run is None else run.spec


async def stamp_manifest_promotion(
    job_id: str,
    *,
    triton_name: str,
    promoted_at: str,
    force_used: bool = False,
    gate_report: dict[str, Any] | None = None,
) -> bool:
    """Set ``manifest.promoted_to`` after a successful Triton handoff.

    ``force_used`` + ``gate_report`` record whether the promote gate was bypassed and what it
    found, so a forced promote is traceable from the manifest alone.

    ``False`` when the run has no manifest yet (a legitimate, non-error case); a database error
    propagates so the caller can tell "nothing to stamp" from "stamping broke".
    """
    run_id = parse_run_id(job_id)
    promoted_to = {
        'triton_model_name': triton_name,
        'promoted_at': promoted_at,
        'force_used': force_used,
        'gate_report': gate_report,
    }
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        return await session.run_sync(
            lambda sync: runs_repo.stamp_promotion(sync, run_id, promoted_to)
        )

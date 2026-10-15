"""Reading a finished run's manifest, run directory, artifacts and job spec,
and stamping a promotion back onto the manifest.

Split out of ``jobs.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.services.training.job_files import (
    _job_path,
    _manifest_path,
    _read_json,
    _status_path,
    _validate_job_id,
)
from src.services.training.job_wire import (
    RUN_ARTIFACT_WHITELIST,
    _public_mlflow_url,
    _rewrite_eval_for_wire,
)


logger = get_logger(__name__)


async def read_manifest(job_id: str) -> dict[str, Any] | None:
    """Return the parsed run manifest or None if absent.

    The trainer writes the manifest at job-end. The promote endpoint reads
    + updates the ``promoted_to`` field so callers can trace which Triton
    model name a checkpoint shipped under. The served ``results.eval`` /
    ``results.mlflow_run_url`` go through the same wire rewrite as
    ``TrainJobStatus`` (see ``_rewrite_eval_for_wire``/``_public_mlflow_url``)
    -- the manifest is a raw dict, not that model, so it isn't covered by
    ``_prepare_status_for_wire`` automatically.
    """
    _validate_job_id(job_id)
    path = _manifest_path(job_id)

    def _do() -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    manifest = await asyncio.to_thread(_do)
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


async def resolve_run_dir(job_id: str) -> Path | None:
    """Resolve the on-disk training-run directory for ``job_id``, best-effort.

    Used only by :func:`read_artifact` to locate metrics/plot files a
    finished (or partially finished) run wrote -- never exposed on the wire
    itself. Derived from the *raw* (unrewritten) ``status.json``:
    ``checkpoint_path`` (``<run_dir>/weights/best.pt``) first, falling back
    to ``eval.confusion_matrix_path`` for a run that failed before export.
    ``None`` if neither is on disk yet.
    """
    _validate_job_id(job_id)
    raw = await _read_json(_status_path(job_id))
    if raw is None:
        return None
    ckpt = raw.get('checkpoint_path')
    if isinstance(ckpt, str) and ckpt:
        return Path(ckpt).resolve().parent.parent
    eval_block = raw.get('eval')
    if isinstance(eval_block, dict):
        cm_path = eval_block.get('confusion_matrix_path')
        if isinstance(cm_path, str) and cm_path:
            return Path(cm_path).resolve().parent
    return None


async def read_artifact(job_id: str, name: str) -> Path | None:
    """Return the absolute path of a whitelisted run artifact, or ``None``.

    ``None`` covers every "don't serve this" case alike (name not in
    :data:`RUN_ARTIFACT_WHITELIST`, unresolvable run dir, missing file, or a
    resolved path that escaped the run dir) -- the router turns any of them
    into a plain 404 so a probe can't learn more than "nothing there."
    """
    _validate_job_id(job_id)
    if name not in RUN_ARTIFACT_WHITELIST:
        return None
    run_dir = await resolve_run_dir(job_id)
    if run_dir is None:
        return None

    def _do() -> Path | None:
        candidate = (run_dir / name).resolve()
        if candidate.parent != run_dir or not candidate.is_file():
            return None
        return candidate

    return await asyncio.to_thread(_do)


async def read_job_spec(job_id: str) -> dict[str, Any] | None:
    """Return the raw ``<job_id>.job.json`` dict, or None if absent.

    Unlike :func:`read_manifest` (trainer-written, best-effort, only exists
    after a run finishes), ``job.json`` is written synchronously by
    :func:`write_job` and always available once a job has been submitted.
    Used by promote to find the registry snapshot pinned at submit time,
    since older/failed runs may lack a manifest entirely.
    """
    _validate_job_id(job_id)
    return await _read_json(_job_path(job_id))


async def stamp_manifest_promotion(
    job_id: str,
    *,
    triton_name: str,
    promoted_at: str,
    force_used: bool = False,
    gate_report: dict[str, Any] | None = None,
) -> bool:
    """Set ``manifest.promoted_to`` after a successful Triton handoff.

    ``force_used`` + ``gate_report`` record whether the promote gate was
    bypassed and, if so, what it found, so a forced promote is traceable
    from the manifest alone.

    Returns False if the manifest doesn't exist yet (older runs without
    one) -- a legitimate, non-error case. Raises on a real failure (corrupt
    JSON, disk/permission error) instead of swallowing it -- the caller
    (the promote router) distinguishes "nothing to stamp" (False, no
    exception) from "stamping broke" (exception) and surfaces the latter
    loudly instead of a silent warn-and-forget.
    """
    _validate_job_id(job_id)
    path = _manifest_path(job_id)

    def _do() -> bool:
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            msg = f'manifest for job {job_id!r} is not a JSON object'
            raise ValueError(msg)
        payload['promoted_to'] = {
            'triton_model_name': triton_name,
            'promoted_at': promoted_at,
            'force_used': force_used,
            'gate_report': gate_report,
        }
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        tmp.replace(path)
        return True

    return await asyncio.to_thread(_do)

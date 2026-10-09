"""Serve-time rewrites for training status/manifest payloads.

Split out of :mod:`src.services.training.jobs`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.services.training.job_models import TrainJobStatus


# GET {api_prefix}/train/artifacts/{job_id}/{name} -- exact basenames only
# (no path traversal is even expressible: no ``/`` is a valid character in
# any of these). Metrics/plots only -- deliberately excludes Ultralytics'
# ``train_batch*.jpg``/``val_batch*.jpg`` (actual training/validation
# images, not aggregate metrics) and ``labels.jpg`` (a dataset-content
# visualization), any of which could leak imagery a deployment doesn't want
# served over this route.
RUN_ARTIFACT_WHITELIST = frozenset(
    {
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'results.png',
        'results.csv',
        'BoxF1_curve.png',
        'BoxP_curve.png',
        'BoxPR_curve.png',
        'BoxR_curve.png',
    }
)

_ARTIFACT_MEDIA_TYPES = {
    '.png': 'image/png',
    '.csv': 'text/csv',
}


# =============================================================================
# Wire rewriting -- internal hostnames / server filesystem paths never reach
# the wire. Applied to every ``TrainJobStatus`` returned by ``read_status``/
# ``list_runs`` and to the raw manifest dict returned by ``read_manifest``.
# =============================================================================


def _public_mlflow_url(run_id: str | None, experiment_id: str | None) -> str | None:
    """Rebuild a browser-reachable MLflow run URL, or ``None``.

    The trainer only knows ``MLFLOW_TRACKING_URI``, a container hostname
    (e.g. ``http://curation-mlflow:5000``) a browser can never resolve.
    ``None`` whenever ``CurationConfig.mlflow_public_url`` is unset, or
    ``run_id``/``experiment_id`` (needed to build the deep-link path)
    aren't both available -- this never falls back to the internal URL.
    """
    if not run_id or not experiment_id:
        return None
    from src.services.resource_links import service_url

    base = service_url('mlflow')
    if not base:
        return None
    return f'{base.rstrip("/")}/#/experiments/{experiment_id}/runs/{run_id}'


def artifact_media_type(name: str) -> str:
    """Content type for a whitelisted run-artifact filename."""
    return _ARTIFACT_MEDIA_TYPES.get(Path(name).suffix.lower(), 'application/octet-stream')


def _artifact_url(job_id: str, artifact_name: str) -> str:
    from src.config.project_context import project_api_base

    return f'{project_api_base()}/train/artifacts/{job_id}/{artifact_name}'


def _rewrite_eval_for_wire(eval_block: Any, job_id: str) -> Any:
    """Replace ``eval.confusion_matrix_path`` (a server filesystem path)
    with ``eval.confusion_matrix_url`` (this route), or ``None``.

    Returns ``eval_block`` unchanged when it isn't a dict (``None``, or a
    validation artifact from a badly-shaped status write).
    """
    if not isinstance(eval_block, dict):
        return eval_block
    out = dict(eval_block)
    cm_path = out.pop('confusion_matrix_path', None)
    out['confusion_matrix_url'] = (
        _artifact_url(job_id, Path(cm_path).name) if isinstance(cm_path, str) and cm_path else None
    )
    return out


def _prepare_status_for_wire(s: TrainJobStatus) -> TrainJobStatus:
    """Apply every serve-time rewrite to a ``TrainJobStatus`` before it's returned."""
    return s.model_copy(
        update={
            'eval': _rewrite_eval_for_wire(s.eval, s.job_id),
            'mlflow_run_url': _public_mlflow_url(s.mlflow_run_id, s.mlflow_experiment_id),
        }
    )

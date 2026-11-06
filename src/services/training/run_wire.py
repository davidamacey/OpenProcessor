"""A ``training_runs`` row plus its ``jobs`` row, shaped as the ``TrainJobStatus`` the routes serve.

The job row carries the live state (status, epoch progress in ``progress``, the latest metrics
and run ids in ``progress.counters``); once the run finished, the manifest in the run row is the
record and wins. Wire rewrites (public MLflow URL, artifact URL for the confusion matrix) are
applied last, in :func:`src.services.training.job_wire._prepare_status_for_wire`.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

from src.domain import job_states as js
from src.domain.training_keys import SKIPPED_ERROR_CODE
from src.services.training.job_models import TrainJobStatus
from src.services.training.job_wire import _prepare_status_for_wire


if TYPE_CHECKING:
    from src.db.repos.training_runs import RunRow

_LIVE_STAGES = ('starting', 'running', 'exporting')


def _iso(value: dt.datetime | None) -> str | None:
    return None if value is None else value.astimezone(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def wire_state(status: str, stage: str | None, error_code: str | None) -> str:
    """The training state vocabulary of the wire from a job status and its progress stage."""
    if status == js.QUEUED:
        return 'queued'
    if status in (js.COMPLETED, js.COMPLETED_WITH_ERRORS):
        return 'finished'
    if status == js.FAILED:
        return 'failed'
    if status == js.CANCELLED:
        return 'skipped' if error_code == SKIPPED_ERROR_CODE else 'cancelled'
    if status == js.INTERRUPTED:
        return 'lost'
    return stage if stage in _LIVE_STAGES else 'running'


def to_status(run: RunRow) -> TrainJobStatus:
    progress = run.progress
    counters: dict[str, Any] = dict(progress.get('counters') or {})
    results: dict[str, Any] = dict((run.manifest or {}).get('results') or {})
    live = {**counters, **{k: v for k, v in results.items() if v is not None}}
    state = wire_state(run.status, progress.get('stage'), run.error_code)
    status = TrainJobStatus(
        job_id=str(run.job_id),
        campaign_id=run.params.get('campaign_id'),
        state=state,
        started_at=_iso(run.started_at),
        finished_at=_iso(run.finished_at),
        current_epoch=progress.get('done'),
        total_epochs=progress.get('total'),
        epoch_time_s=live.get('epoch_time_s'),
        last_epoch_metric=live.get('last_epoch_metric'),
        best_checkpoint_metric=live.get('best_checkpoint_metric'),
        mlflow_run_id=live.get('mlflow_run_id') or run.mlflow_run_id,
        mlflow_experiment_id=live.get('mlflow_experiment_id'),
        checkpoint_path=live.get('checkpoint_path'),
        gpu=list(counters.get('gpu') or []),
        eval=live.get('eval'),
        compare=live.get('compare'),
        error=run.error_message,
        heartbeat_at=_iso(run.heartbeat_at),
        class_remap_copy_failed=bool(counters.get('class_remap_copy_failed', False)),
    )
    return _prepare_status_for_wire(status)

"""Ultralytics hooks: cancel sentinel, epoch metric capture and the MLflow artifact-root guard."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from logutil import get_logger


if TYPE_CHECKING:
    from job_spec import JobSpec
    from job_status import StatusState


logger = get_logger('trainer.callbacks')


class CancelRequestedError(Exception):
    """Raised inside the Ultralytics callback when a ``.cancel`` sentinel appears."""


def _make_ultralytics_callbacks(
    spec: JobSpec,
    state: StatusState,
    total_epochs: int,
) -> tuple[Any, Any, Any]:
    """Build the three per-epoch callbacks registered on the model.

    They read the cancel sentinel between epochs and refresh ``state`` so the
    heartbeat thread publishes live metrics.
    """
    epoch_start_at: dict[str, float] = {}

    def on_train_epoch_start(_trainer: Any) -> None:
        epoch_start_at['t0'] = time.time()

    def on_train_epoch_end(_trainer: Any) -> None:
        if spec.cancel_path.exists():
            logger.info('cancel sentinel detected -- stopping after epoch', job_id=spec.job_id)
            msg = f'cancel sentinel for {spec.job_id}'
            raise CancelRequestedError(msg)

    def on_fit_epoch_end(trainer: Any) -> None:
        """Capture per-epoch metrics -- and tell the true last training
        epoch apart from Ultralytics' post-training re-validation of
        best.pt.

        Ultralytics' ``BaseTrainer.final_eval()`` re-validates the best
        checkpoint after training completes and fires this same callback
        one more time -- but does not advance ``trainer.epoch``. So a
        repeated epoch number (same as the previous call) is the signal
        that this call is the best checkpoint's own metrics, not a new
        training epoch; only that call updates ``best_checkpoint_metric``.
        Every other call is a real epoch and updates ``last_epoch_metric``.
        This keeps both as one coherent row (map50 + map50_95 from the SAME
        validation pass) instead of a per-key running max that can mix
        metrics from different epochs.
        """
        try:
            epoch = int(getattr(trainer, 'epoch', 0)) + 1  # Ultralytics is 0-indexed
            t0 = epoch_start_at.pop('t0', None)
            epoch_dt = (time.time() - t0) if t0 is not None else None
            metrics = getattr(trainer, 'metrics', None) or {}
            map50 = metrics.get('metrics/mAP50(B)') or metrics.get('metrics/mAP_0.5')
            map5095 = metrics.get('metrics/mAP50-95(B)') or metrics.get('metrics/mAP_0.5:0.95')
            row: dict[str, Any] = {'epoch': epoch}
            if map50 is not None:
                row['map50'] = float(map50)
            if map5095 is not None:
                row['map50_95'] = float(map5095)
            with state.lock:
                is_best_checkpoint_revalidation = (
                    state.last_epoch_metric is not None
                    and state.last_epoch_metric.get('epoch') == epoch
                )
                if is_best_checkpoint_revalidation:
                    state.best_checkpoint_metric = row
                else:
                    state.current_epoch = epoch
                    state.total_epochs = total_epochs
                    state.epoch_time_s = epoch_dt
                    state.last_epoch_metric = row
        except Exception as exc:  # metric capture must not kill the epoch
            logger.warning('on_fit_epoch_end metric capture failed', error=str(exc))

    return on_train_epoch_start, on_train_epoch_end, on_fit_epoch_end


# ---------------------------------------------------------------------------
# ML-1 defense in depth
# ---------------------------------------------------------------------------


def _guard_ultralytics_mlflow_artifact_root(experiment_name: str) -> None:
    """Disable Ultralytics' own MLflow integration if its artifact root
    is a local path this process cannot write.

    The primary fix is serving MLflow artifacts through the tracking
    server's HTTP proxy (``--serve-artifacts
    --default-artifact-root=mlflow-artifacts:/``, see ``docker-compose.yml``).
    This guard is a second line of defense against a misconfigured or
    stale experiment whose ``artifact_location`` is still a bare local
    path (e.g. an experiment created before that fix, or a deployment
    that reverts it): Ultralytics' own ``on_train_end`` MLflow callback
    (``ultralytics/utils/callbacks/mlflow.py``) has no try/except around
    ``mlflow.log_artifact(...)``, so a PermissionError there escapes
    ``model.train()`` and fails an entire multi-hour run at the last
    step, after every epoch already succeeded. The project's own
    callbacks in ``mlflow_callbacks.py`` are unaffected -- they already
    catch their own artifact-logging failures.
    """
    try:
        import importlib

        import mlflow

        ultralytics_settings = importlib.import_module('ultralytics.utils').SETTINGS
    except Exception as exc:  # pragma: no cover - optional dependency wiring
        logger.warning('mlflow artifact-root guard: import failed', error=str(exc))
        return

    if not ultralytics_settings.get('mlflow', False):
        return  # Ultralytics' own integration isn't enabled; nothing to guard.

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return  # not created yet; the server will apply --default-artifact-root
        location = experiment.artifact_location or ''
        is_local_path = location.startswith(('/', 'file:'))
        if not is_local_path:
            return  # proxied (mlflow-artifacts:) or remote (s3:, gs:, ...) -- fine
        probe_dir = Path(location.removeprefix('file:'))
        probe_dir.mkdir(parents=True, exist_ok=True)
        test_file = probe_dir / '.write_probe'
        test_file.write_text('ok')
        test_file.unlink()
    except Exception as exc:
        logger.warning(
            "mlflow artifact root is unwritable; disabling Ultralytics' built-in "
            "MLflow integration for this run (the project's own mlflow_callbacks.py "
            'still records params/metrics)',
            artifact_location=location if 'location' in locals() else None,
            error=str(exc),
        )
        ultralytics_settings.update({'mlflow': False})

"""Status file: :class:`StatusState`, the ``status.json`` payload, heartbeats and MLflow run-id capture.

See :mod:`job_protocol` for the overall file protocol.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from job_spec import JobSpec, _utcnow_iso
from logutil import get_logger


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger('trainer.job_status')


HEARTBEAT_INTERVAL_S = 5.0
TERMINAL_STATES = {'finished', 'failed', 'cancelled', 'skipped'}
ACTIVE_STATES = {'queued', 'starting', 'running', 'exporting'}


@dataclass
class StatusState:
    """Mutable state shared between the trainer thread and heartbeat thread."""

    job_id: str
    campaign_id: str | None = None
    state: str = 'queued'
    started_at: str | None = None
    finished_at: str | None = None
    current_epoch: int = 0
    total_epochs: int = 0
    epoch_time_s: float | None = None
    # The true last TRAINING epoch's metrics (with its epoch number) --
    # distinct from best_checkpoint_metric because Ultralytics'
    # final_eval() fires the same on_fit_epoch_end callback once more
    # after training, re-validating best.pt, and that call does not
    # advance trainer.epoch -- see docker/trainer/trainer.py's
    # on_fit_epoch_end for how the two are told apart.
    last_epoch_metric: dict[str, Any] | None = None
    # The best checkpoint's (best.pt) own re-validation metrics, as one
    # coherent row (both map50 and map50_95 from the SAME validation
    # pass) -- not a per-key running max across every epoch, which can
    # mix map50 from one epoch with map50_95 from another.
    best_checkpoint_metric: dict[str, Any] | None = None
    mlflow_run_id: str | None = None
    mlflow_run_url: str | None = None
    # Needed alongside ``mlflow_run_id`` for the API to rebuild a
    # browser-reachable deep link (see ``src/services/training/job_wire.py``'s
    # ``_public_mlflow_url``) -- the run URL's path embeds both ids and the
    # API has no other way to recover the experiment id.
    mlflow_experiment_id: str | None = None
    checkpoint_path: str | None = None
    eval: dict[str, Any] | None = None
    compare: dict[str, Any] | None = None
    error: str | None = None
    # Actual seed/deterministic values handed to Ultralytics' model.train(),
    # populated once train_kwargs is assembled. ``None`` until then (e.g. a job
    # that fails before reaching training). The manifest reads these rather
    # than re-deriving from the spec, so it records what actually ran.
    actual_seed: int | None = None
    actual_deterministic: bool | None = None
    # True when class_remap.json existed in the tmp subset dir but the copy
    # into the checkpoint's weights/ dir failed. Surfaced on the status payload
    # so promote can refuse to silently fall back to the full registry for a
    # subset-trained run.
    class_remap_copy_failed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` atomically (tmp file + rename).

    The API polls these files; a half-written status must never be readable.
    """
    # Unique per writer: the heartbeat thread and the main thread both write the
    # same status file, and a shared tmp name let one rename the other's away.
    tmp = path.with_suffix(f'{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Status / heartbeat
# ---------------------------------------------------------------------------


def _host_gpu_order() -> tuple[int, ...]:
    """``OP_TRAIN_GPU_ORDER`` (the compose ``device_ids`` list); empty = unrestricted."""
    raw = os.environ.get('OP_TRAIN_GPU_ORDER', '')
    return tuple(int(tok) for tok in raw.split(',') if tok.strip())


def _gpu_telemetry() -> list[dict[str, Any]]:
    """Read util / memory from each visible GPU via NVML.

    Returns ``[]`` on hosts without NVML (CI, CPU smoke runs).
    """
    try:
        import pynvml
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:  # NVML raises its own error family
        return []
    out: list[dict[str, Any]] = []
    order = _host_gpu_order()
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            except Exception as exc:  # one bad GPU must not blank the rest
                logger.warning('nvml: per-GPU read failed', index=i, error=str(exc))
                continue
            out.append(
                {
                    # NVML indexes are container-local; report the host id the API reasons in.
                    'index': order[i] if i < len(order) else i,
                    'util_pct': int(util.gpu),
                    'mem_used_mb': int(mem.used // (1024 * 1024)),
                    'mem_total_mb': int(mem.total // (1024 * 1024)),
                }
            )
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
    return out


def build_status_payload(s: StatusState) -> dict[str, Any]:
    """Render the ``status.json`` body.

    Every key here is read by :class:`src.services.training.job_models.TrainJobStatus`
    (which allows extras, so adding a key is safe; removing one is not).
    """
    with s.lock:
        return {
            'job_id': s.job_id,
            'campaign_id': s.campaign_id,
            'state': s.state,
            'started_at': s.started_at,
            'finished_at': s.finished_at,
            'current_epoch': s.current_epoch,
            'total_epochs': s.total_epochs,
            'epoch_time_s': s.epoch_time_s,
            'last_epoch_metric': s.last_epoch_metric,
            'best_checkpoint_metric': s.best_checkpoint_metric,
            'mlflow_run_id': s.mlflow_run_id,
            'mlflow_run_url': s.mlflow_run_url,
            'mlflow_experiment_id': s.mlflow_experiment_id,
            'checkpoint_path': s.checkpoint_path,
            'class_remap_copy_failed': s.class_remap_copy_failed,
            'gpu': _gpu_telemetry(),
            'eval': s.eval,
            'compare': s.compare,
            'error': s.error,
            'heartbeat_at': _utcnow_iso(),
        }


def write_status_now(status_path: Path, state: StatusState) -> None:
    """Force-write the current status (used at state transitions)."""
    _atomic_write_json(status_path, build_status_payload(state))


def _heartbeat_loop(stop_event: threading.Event, state: StatusState, status_path: Path) -> None:
    """Daemon-thread body: write status.json every ``HEARTBEAT_INTERVAL_S``.

    The API flips a run to ``lost`` when ``heartbeat_at`` goes stale, so this
    loop must keep running for the whole job.
    """
    while not stop_event.is_set():
        try:
            _atomic_write_json(status_path, build_status_payload(state))
        except OSError:
            logger.exception('heartbeat: write failed', path=str(status_path))
        # Sleep in small increments so a stop is honored quickly.
        for _ in range(int(HEARTBEAT_INTERVAL_S * 10)):
            if stop_event.is_set():
                break
            time.sleep(0.1)


def _bound_mlflow_http_waits() -> None:
    """Cap MLflow's HTTP timeout + retries before any tracking call.

    MLflow's default client retries with exponential backoff for minutes. This
    runs on the path between "training finished" and "status says finished", so
    an unreachable tracking server would otherwise hold a completed run in
    ``exporting`` long enough for the API to declare the heartbeat lost.
    ``setdefault`` keeps an operator override winning.
    """
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT', '10')
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '2')
    os.environ.setdefault('MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR', '1')


def _capture_mlflow_run_id(spec: JobSpec, state: StatusState) -> None:
    """Record this job's MLflow run id + deep-link URL into the status.

    Best-effort. Ultralytics names the run after ``train_kwargs['name']`` (==
    ``spec.mlflow_run_name``) and logs under ``MLFLOW_EXPERIMENT_NAME``. We
    search the tracking server by name rather than reading
    ``mlflow.active_run()``: the run is closed by the time training returns and,
    under DDP, lives in a spawned subprocess this process can't see.
    """
    _bound_mlflow_http_waits()
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'openprocessor')
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.run_name = '{spec.mlflow_run_name}'",
            order_by=['attributes.start_time DESC'],
            max_results=1,
        )
        if not runs:
            return
        run = runs[0]
        with state.lock:
            state.mlflow_run_id = run.info.run_id
            state.mlflow_experiment_id = str(experiment.experiment_id)
            # Internal-network URL -- a browser behind the deployment can't
            # reach ``tracking_uri`` (it's a container hostname). Kept here
            # for operators inspecting status.json/manifest.json directly
            # (e.g. ``docker exec``); the API rewrites this to a public URL
            # (or null) before it ever reaches the wire -- see
            # ``src/services/training/job_wire.py``'s ``_public_mlflow_url``.
            state.mlflow_run_url = (
                f'{tracking_uri.rstrip("/")}/#/experiments/'
                f'{experiment.experiment_id}/runs/{run.info.run_id}'
            )
    except Exception as exc:  # tracking is optional
        logger.warning('mlflow run-id capture failed', error=str(exc))

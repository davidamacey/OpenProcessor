"""The API <-> trainer job-file protocol, trainer side.

The API never RPCs the trainer: the entire contract is files in a shared
volume. The API-side half lives in :mod:`src.services.training.jobs`; this
module is the other half and must stay wire-compatible with it.

    API     writes  <jobs_dir>/<job_id>.job.json       to start a run
    trainer writes  <jobs_dir>/<job_id>.status.json    while it runs
    API     writes  <jobs_dir>/<job_id>.cancel         to cancel
    trainer appends <jobs_dir>/<job_id>.run.log        for the log-tail route
    trainer writes  <jobs_dir>/<job_id>.manifest.json  at the end

Everything here is about those files -- parsing a submitted job, publishing
status + heartbeats, tee-ing the run log, writing the lineage manifest, and
deciding which jobs are still pending. The training loop itself lives in
:mod:`trainer`, dataset preparation in :mod:`dataset_prep`, and cross-job
campaign policy in :mod:`campaign`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from logutil import get_logger


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = get_logger('trainer.job_protocol')


# ---------------------------------------------------------------------------
# Constants + environment
# ---------------------------------------------------------------------------

HEARTBEAT_INTERVAL_S = 5.0
CANCEL_FILENAME_SUFFIX = '.cancel'

# Mirrors src.services.training.jobs -- keep both in sync.
VALID_FAMILIES = {'yolo26'}
VALID_SIZES = {'n', 's', 'm', 'l', 'x'}
VALID_PROFILES = {'probe', 'nano', 'small', 'medium', 'large', 'xlarge', 'custom'}
TERMINAL_STATES = {'finished', 'failed', 'cancelled', 'skipped'}
ACTIVE_STATES = {'queued', 'starting', 'running', 'exporting'}

# Shared job directory. Same default + env var as the API side
# (``src.services.training.jobs._resolve_jobs_dir``).
DEFAULT_JOBS_DIR = Path(os.environ.get('OP_TRAIN_JOBS_DIR', '/jobs'))

# Scratch root for per-job subset/augmentation rewrites. Inside the container
# this is the image's own tmpfs, not a shared host /tmp, and every job writes
# into its own ``<TMP_ROOT>/<job_id>/`` subdirectory which ``run_job`` rmtree's
# on exit. Overridable via OP_TRAIN_TMP_ROOT for a deployment that wants the
# scratch space on a larger volume.
TMP_ROOT = Path(os.environ.get('OP_TRAIN_TMP_ROOT', '/tmp'))  # nosec B108 - container tmpfs


# Ultralytics' ONNX export batch cap. Matches
# src.services.training.yolo_triton_config.DEFAULT_MAX_BATCH so a promote with
# default settings serves an engine the export actually supports.
DEFAULT_EXPORT_BATCH = 8
DEFAULT_INPUT_SIZE = 640


# ---------------------------------------------------------------------------
# Job + status dataclasses
# ---------------------------------------------------------------------------


@dataclass
class JobSpec:
    """Parsed + validated ``job.json``.

    Field-for-field a subset of :class:`src.services.training.jobs.TrainJobSpec`
    -- unknown keys are tolerated (kept in :attr:`raw`) so an API that grows a
    field doesn't break an older trainer image.
    """

    job_id: str
    job_path: Path
    model_family: str
    model_size: str
    profile: str
    dataset_export_dir: Path
    include_classes: list[int] | None
    single_cls: bool
    cuda_visible_devices: str | None
    hyperparameters: dict[str, Any]
    augmentation: dict[str, Any]
    mlflow_run_name: str
    submitted_at: str
    # Campaign metadata. Only populated for jobs written by
    # ``POST {api_prefix}/train/start_campaign``.
    campaign_id: str | None = None
    stop_when: dict[str, float] | None = None
    auto_promote_best: bool = False
    is_last_in_campaign: bool = False
    # Opt-in: on a finished run, drop a bake-off job that exports this
    # checkpoint to portable ONNX (fp32/fp16/int8) and benchmarks
    # size/speed/accuracy, so a quantization panel updates with no manual step.
    auto_quantize_bakeoff: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def status_path(self) -> Path:
        return self.job_path.parent / f'{self.job_id}.status.json'

    @property
    def run_log_path(self) -> Path:
        return self.job_path.parent / f'{self.job_id}.run.log'

    @property
    def manifest_path(self) -> Path:
        return self.job_path.parent / f'{self.job_id}.manifest.json'

    @property
    def cancel_path(self) -> Path:
        return self.job_path.parent / f'{self.job_id}{CANCEL_FILENAME_SUFFIX}'

    @property
    def tmp_root(self) -> Path:
        return TMP_ROOT / self.job_id

    @property
    def subset_dir(self) -> Path:
        return self.tmp_root / 'subset'

    @property
    def is_subset_run(self) -> bool:
        return bool(self.include_classes) or self.single_cls


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
    best_metric: dict[str, float] | None = None
    last_metric: dict[str, float] | None = None
    mlflow_run_id: str | None = None
    mlflow_run_url: str | None = None
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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def _read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        msg = f'{path} is not a JSON object'
        raise ValueError(msg)
    return loaded


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` atomically (tmp file + rename).

    The API polls these files; a half-written status must never be readable.
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)
    tmp.replace(path)


def _status_path_for(job_path: Path) -> Path:
    """Sibling ``<job_id>.status.json`` for a ``<job_id>.job.json`` path."""
    return job_path.parent / f'{job_path.name[: -len(".job.json")]}.status.json'


def _job_id_from_path(job_path: Path) -> str:
    return job_path.name[: -len('.job.json')]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def parse_and_validate_job(job_path: Path) -> JobSpec:
    """Read ``job.json`` and reject obviously-broken submissions.

    Most preflight checks (free disk, class counts, ...) live in the API's
    ``POST {api_prefix}/train/preflight``. The trainer only re-checks the few
    constraints that protect the training loop itself, so a hand-written or
    stale job file can't wedge the watcher.
    """
    raw = _read_json(job_path)

    job_id = str(raw.get('job_id') or '')
    if not job_id:
        msg = "job.json missing 'job_id'"
        raise ValueError(msg)

    family = raw.get('model_family') or 'yolo26'
    if family not in VALID_FAMILIES:
        msg = f'unsupported model_family={family!r}; expected one of {sorted(VALID_FAMILIES)}'
        raise ValueError(msg)

    size = raw.get('model_size') or 'm'
    if size not in VALID_SIZES:
        msg = f'unsupported model_size={size!r}; expected one of {sorted(VALID_SIZES)}'
        raise ValueError(msg)

    profile = raw.get('profile') or 'medium'
    if profile not in VALID_PROFILES:
        msg = f'unsupported profile={profile!r}; expected one of {sorted(VALID_PROFILES)}'
        raise ValueError(msg)

    export_dir_raw = raw.get('dataset_export_dir')
    if not export_dir_raw:
        msg = "job.json missing 'dataset_export_dir'"
        raise ValueError(msg)
    export_dir = Path(export_dir_raw)
    if not export_dir.is_dir():
        msg = f'dataset_export_dir does not exist: {export_dir}'
        raise ValueError(msg)

    include_classes = raw.get('include_classes')
    if include_classes is not None and (
        not isinstance(include_classes, list)
        or not all(isinstance(c, int) for c in include_classes)
    ):
        msg = 'include_classes must be a list[int]'
        raise ValueError(msg)

    hyperparameters = dict(raw.get('hyperparameters') or {})
    # YOLO26 + optimizer=auto silently trades MuSGD for AdamW on short runs
    # (Ultralytics issue #23696). The API rejects it too; re-checked here in
    # case a job file was written by hand.
    if family == 'yolo26' and str(hyperparameters.get('optimizer', '')).lower() == 'auto':
        msg = (
            "optimizer='auto' is rejected for YOLO26 (Ultralytics issue #23696); "
            'set optimizer=MuSGD explicitly'
        )
        raise ValueError(msg)

    cuda_visible = raw.get('cuda_visible_devices')
    stop_when_raw = raw.get('stop_when')
    stop_when = (
        {str(k): float(v) for k, v in stop_when_raw.items()}
        if isinstance(stop_when_raw, dict)
        else None
    )
    campaign_id = raw.get('campaign_id')

    return JobSpec(
        job_id=job_id,
        job_path=job_path,
        model_family=family,
        model_size=size,
        profile=profile,
        dataset_export_dir=export_dir,
        include_classes=include_classes,
        single_cls=bool(raw.get('single_cls', False)),
        cuda_visible_devices=str(cuda_visible) if cuda_visible else None,
        hyperparameters=hyperparameters,
        augmentation=dict(raw.get('augmentation') or {}),
        mlflow_run_name=raw.get('mlflow_run_name') or job_id,
        submitted_at=raw.get('submitted_at') or _utcnow_iso(),
        campaign_id=str(campaign_id) if campaign_id else None,
        stop_when=stop_when,
        auto_promote_best=bool(raw.get('auto_promote_best', False)),
        is_last_in_campaign=bool(raw.get('is_last_in_campaign', False)),
        auto_quantize_bakeoff=bool(raw.get('auto_quantize_bakeoff', False)),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Status / heartbeat
# ---------------------------------------------------------------------------


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
                    'index': i,
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

    Every key here is read by :class:`src.services.training.jobs.TrainJobStatus`
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
            'best_metric': s.best_metric,
            'last_metric': s.last_metric,
            'mlflow_run_id': s.mlflow_run_id,
            'mlflow_run_url': s.mlflow_run_url,
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
            state.mlflow_run_url = (
                f'{tracking_uri.rstrip("/")}/#/experiments/'
                f'{experiment.experiment_id}/runs/{run.info.run_id}'
            )
    except Exception as exc:  # tracking is optional
        logger.warning('mlflow run-id capture failed', error=str(exc))


# ---------------------------------------------------------------------------
# Run-log tee
# ---------------------------------------------------------------------------


@contextmanager
def tee_to_run_log(run_log_path: Path) -> Iterator[None]:
    """Tee stdout + stderr into ``run_log_path``.

    This is what makes ``GET {api_prefix}/train/log/tail/{job_id}`` show
    Ultralytics' own progress output.
    """
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = run_log_path.open('a', encoding='utf-8', buffering=1)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    class _Tee:
        def __init__(self, *streams: Any) -> None:
            self._streams = streams

        def write(self, data: str) -> int:
            for s in self._streams:
                with contextlib.suppress(Exception):
                    s.write(data)
                    s.flush()
            return len(data)

        def flush(self) -> None:
            for s in self._streams:
                with contextlib.suppress(Exception):
                    s.flush()

        def isatty(self) -> bool:
            return False

    try:
        sys.stdout = _Tee(orig_stdout, fp)  # type: ignore[assignment]
        sys.stderr = _Tee(orig_stderr, fp)  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        with contextlib.suppress(Exception):
            fp.close()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _checkpoint_sha256(checkpoint_path: Path) -> str | None:
    """Return the sha256 hex digest of ``checkpoint_path``, or ``None``."""
    if not checkpoint_path.is_file():
        return None
    try:
        h = hashlib.sha256()
        with checkpoint_path.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                h.update(chunk)
    except OSError as exc:
        logger.warning('checkpoint sha256 failed', path=str(checkpoint_path), error=str(exc))
        return None
    return h.hexdigest()


def _ultralytics_version() -> str | None:
    """Best-effort read of the installed ultralytics package version."""
    try:
        import importlib.metadata as md

        return md.version('ultralytics')
    except Exception:  # PackageNotFoundError family
        return None


def build_lineage(spec: JobSpec) -> dict[str, Any]:
    """Byte-identical lineage dict shared by the run manifest and the
    MLflow tags/params/registry metadata (``mlflow_callbacks.py``).

    Reads straight off ``spec.raw`` -- the job.json the API wrote via
    ``src.services.training.jobs.write_job`` -- so both consumers see the
    exact same values with no second source of truth to drift out of sync.
    ``trainer_sha`` prefers this container's own baked ``OP_BUILD_SHA``
    (the trainer image actually running this code) over the API's stamped
    ``trainer_image_revision`` (the trainer image the API *observed* at
    submit time, via the docker socket) -- the two agree unless the
    trainer image was rebuilt/redeployed between submit and run.
    """
    raw = spec.raw
    return {
        'dataset_sha': raw.get('dataset_sha'),
        'frozen_test_sha': raw.get('frozen_test_sha'),
        'test_label_sha': raw.get('test_label_sha'),
        'dataset_version_tag': raw.get('dataset_version_tag'),
        'api_sha': raw.get('api_sha'),
        'trainer_sha': os.environ.get('OP_BUILD_SHA') or raw.get('trainer_image_revision') or None,
        'trainer_image_id': raw.get('trainer_image_id'),
    }


def write_manifest(
    spec: JobSpec, state: StatusState, class_remap: dict[str, Any] | None = None
) -> None:
    """Write ``<job_id>.manifest.json`` -- the run's full lineage envelope.

    Captures dataset SHA, class remap, code versions and eval results so a
    finished run can be reproduced and audited. Read by
    ``GET {api_prefix}/train/manifest/{job_id}`` and by promote (which resolves
    ``lineage.class_remap`` from here first). Writes are atomic and
    best-effort: a failure logs and moves on rather than blocking the trainer's
    terminal-status write.

    ``class_remap`` is passed in rather than read here: it lives under the
    job's scratch dir, which the caller deletes right after this runs, and this
    module has no business knowing where dataset preparation put it (see
    :func:`dataset_prep.read_class_remap`). ``None`` is correct for a
    whole-export run.
    """
    try:
        checkpoint_sha = (
            _checkpoint_sha256(Path(state.checkpoint_path)) if state.checkpoint_path else None
        )
        lineage = build_lineage(spec)
        manifest: dict[str, Any] = {
            'kind': 'train',
            'job_id': spec.job_id,
            'campaign_id': spec.campaign_id,
            'created_at': datetime.now(tz=UTC).isoformat(),
            'lineage': {
                'export_dir': str(spec.dataset_export_dir),
                'dataset_sha': lineage['dataset_sha'],
                'frozen_test_sha': lineage['frozen_test_sha'],
                'test_label_sha': lineage['test_label_sha'],
                'dataset_version_tag': lineage['dataset_version_tag'],
                'include_classes': spec.include_classes,
                'single_cls': spec.single_cls,
                'class_remap': class_remap,
                'augmentation_seed': (spec.augmentation or {}).get('seed', 42),
                # Actual values handed to model.train(), not just what was
                # requested -- falls back to the requested/default value only
                # if the run failed before train_kwargs was assembled.
                'training_seed': (
                    state.actual_seed
                    if state.actual_seed is not None
                    else int(spec.hyperparameters.get('seed') or 42)
                ),
                'deterministic': (
                    state.actual_deterministic if state.actual_deterministic is not None else True
                ),
                # Registry snapshot the API pinned at submit time -- see
                # src/services/training/jobs.py::write_job.
                'registry_sha': spec.raw.get('registry_sha'),
            },
            'code_versions': {
                'api_sha': lineage['api_sha'],
                'trainer_sha': lineage['trainer_sha'],
                'trainer_image_id': lineage['trainer_image_id'],
                'ultralytics_pkg': _ultralytics_version(),
                'ultralytics_sha': os.environ.get('ULTRALYTICS_SHA'),
            },
            'spec': {
                'model_family': spec.model_family,
                'model_size': spec.model_size,
                'profile': spec.profile,
                'cuda_visible_devices': spec.cuda_visible_devices,
                'hyperparameters': spec.hyperparameters,
                # ``spec.augmentation`` is ALWAYS a dict by this point (the
                # job.json parser coerces a missing/null value to ``{}``), so
                # recording it verbatim would lose the distinction between "no
                # augmentation requested" and "an empty-but-present block". A
                # reproduce flow resubmits this value as-is, and
                # ``AugmentationSpec`` parses a bare ``{}`` into a fully
                # populated, ``enabled=True``-by-default spec -- so a run that
                # never used augmentation would silently get it turned ON.
                # Recording ``None`` for the genuinely-empty case keeps
                # reproduce identical in behavior, not just in JSON shape.
                'augmentation': spec.augmentation or None,
            },
            'results': {
                'final_state': state.state,
                'eval': state.eval,
                'compare': state.compare,
                'best_metric': state.best_metric,
                'checkpoint_path': state.checkpoint_path,
                'checkpoint_sha256': checkpoint_sha,
                'mlflow_run_id': state.mlflow_run_id,
                'mlflow_run_url': state.mlflow_run_url,
            },
            # Stamped by POST {api_prefix}/train/promote/{job_id}.
            'promoted_to': None,
        }
        tmp = spec.manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
        tmp.replace(spec.manifest_path)
    except Exception as exc:  # manifest must never block the status write
        logger.warning('manifest write failed', job_id=spec.job_id, error=str(exc))


# ---------------------------------------------------------------------------
# Work queue: which job files still need running
# ---------------------------------------------------------------------------


def _is_terminal(status_path: Path) -> bool:
    """True when a job's status.json already shows a terminal state."""
    if not status_path.is_file():
        return False
    try:
        return _read_json(status_path).get('state') in TERMINAL_STATES
    except (OSError, ValueError):
        return False


def list_pending_jobs(jobs_dir: Path) -> list[Path]:
    """Find ``*.job.json`` files that haven't reached a terminal state.

    Sorted by mtime so older jobs run first (FIFO), matching the order the API
    writes a campaign's runs in.
    """
    if not jobs_dir.is_dir():
        return []
    return [
        p
        for p in sorted(jobs_dir.glob('*.job.json'), key=lambda x: x.stat().st_mtime)
        if not _is_terminal(_status_path_for(p))
    ]


def reject_job(job_path: Path, exc: Exception) -> None:
    """Write a terminal ``failed`` status for an unparseable job file.

    Without this the watcher would re-read the same broken file forever.
    """
    status_path = _status_path_for(job_path)
    payload = {
        'job_id': _job_id_from_path(job_path),
        'state': 'failed',
        'error': f'validation: {exc}',
        'finished_at': _utcnow_iso(),
        'heartbeat_at': _utcnow_iso(),
    }
    try:
        _atomic_write_json(status_path, payload)
    except OSError:
        logger.exception('failed to write rejection status', path=str(status_path))

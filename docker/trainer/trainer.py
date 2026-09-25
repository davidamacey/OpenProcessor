"""Trainer watcher + training loop -- the trainer container's entrypoint.

Polls the shared jobs directory for ``*.job.json`` files and runs each through
Ultralytics: GPU selection, dataset preparation, the training call (with
cooperative cancel and CUDA-OOM batch backoff), checkpoint + ONNX export, and
metric capture. The file protocol it speaks lives in :mod:`job_protocol`,
dataset preparation in :mod:`dataset_prep`, cross-job campaign policy in
:mod:`campaign`, and the optional comparison against a served model in
:mod:`incumbent_compare`.

Nothing domain-specific is hardcoded. Everything that varies per deployment --
dataset location, class subset, hyperparameters, augmentation, which classes
are orientation-sensitive, which Triton model to compare against -- arrives
through ``job.json`` or the ``OP_*`` environment (see ``env.template``).

Usage::

    python -m trainer --watch /jobs          # daemon (the container ENTRYPOINT)
    python -m trainer --watch /jobs --once   # process one job then exit (tests)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import incumbent_compare
from campaign import maybe_handle_campaign, write_quant_bakeoff_job
from dataset_prep import copy_class_remap_to_weights_dir, prepare_dataset, read_class_remap
from job_protocol import (
    DEFAULT_EXPORT_BATCH,
    DEFAULT_INPUT_SIZE,
    DEFAULT_JOBS_DIR,
    JobSpec,
    StatusState,
    _capture_mlflow_run_id,
    _heartbeat_loop,
    _utcnow_iso,
    build_lineage,
    list_pending_jobs,
    parse_and_validate_job,
    reject_job,
    tee_to_run_log,
    write_manifest,
    write_status_now,
)
from logutil import get_logger


logger = get_logger('trainer')


# Ultralytics' ``project=`` dir for run outputs. This MUST be the same absolute
# path in every container that later reads a path recorded in
# status.json/manifest.json (``checkpoint_path``, the ONNX sibling) -- the API
# container resolves ``checkpoint_path`` verbatim when promoting, so a
# trainer-private mount alias 404s the moment promote runs. Both containers
# mount the same host dir at this path; see docker-compose.yml.
RUNS_ROOT = os.environ.get('OP_TRAIN_RUNS_ROOT', '/var/lib/openprocessor/training_runs')

POLL_INTERVAL_S = 2.0


# Host GPU ids this container is attached to, in the exact order Docker's
# ``deploy.resources.reservations.devices[].device_ids`` lists them.
# Container-local CUDA indices are positional within that list, NOT the host
# ids -- so a container given ``device_ids: ['0', '2']`` sees host GPU 2 as
# local index 1. ``job.json``'s ``cuda_visible_devices`` names HOST ids (the
# API's GPU arbiter reasons in host ids), so they must be translated.
#
# Unset (the default) means the container sees every GPU at its host index and
# the job's device string is passed through unchanged. Set it to the compose
# file's ``device_ids`` list whenever the trainer is pinned to a GPU subset;
# they MUST be kept in sync.
def _gpu_order() -> tuple[int, ...]:
    raw = os.environ.get('OP_TRAIN_GPU_ORDER', '')
    return tuple(int(tok) for tok in raw.split(',') if tok.strip())


# CUDA-OOM backoff: halve the batch and retry with a clean model, so an
# unattended run completes instead of dying on a too-large batch.
MAX_OOM_RETRIES = 5
AUTOBATCH_VRAM_FRACTION = 0.80
OOM_FALLBACK_BATCH_PER_GPU = 16


# ---------------------------------------------------------------------------
# GPU device resolution
# ---------------------------------------------------------------------------


def resolve_local_device(cuda_visible_devices: str | None) -> str | None:
    """Map host-index GPU ids (job spec) to container-local CUDA indices.

    ``spec.cuda_visible_devices`` names *host* GPU slots -- that's the id space
    the API's GPU arbiter reasons in. When the container is pinned to a subset
    of GPUs via Docker ``device_ids``, those appear inside it as local indices
    ``0..n-1`` in the order listed, NOT as their host ids.

    Deriving the local device string from just the *count* of requested GPUs
    (e.g. always ``'0'`` for any single-GPU request) is wrong the moment the
    requested GPU isn't first in :func:`_gpu_order`: a request for host GPU 2
    would silently run on host GPU 0.

    With ``OP_TRAIN_GPU_ORDER`` unset the container sees every GPU at its host
    index, so the string passes through unchanged.

    Raises:
        ValueError: if a requested host GPU isn't one this container is
            attached to -- surfaced as a job failure rather than a silent
            mis-schedule.
    """
    hosts = [int(tok) for tok in (cuda_visible_devices or '').split(',') if tok.strip()]
    if not hosts:
        return None
    order = _gpu_order()
    if not order:
        return ','.join(str(h) for h in hosts)
    locals_: list[str] = []
    for host_id in hosts:
        try:
            locals_.append(str(order.index(host_id)))
        except ValueError:
            msg = (
                f'requested host GPU {host_id} is not attached to this container '
                f'(attached: {order}; set OP_TRAIN_GPU_ORDER to match the compose '
                f'device_ids list)'
            )
            raise ValueError(msg) from None
    return ','.join(locals_)


# ---------------------------------------------------------------------------
# Cancellation + epoch callbacks
# ---------------------------------------------------------------------------


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
# Eval block
# ---------------------------------------------------------------------------


def populate_eval_block(
    state: StatusState,
    save_dir: Path,
    val_results: Any | None = None,
    data_yaml_path: Path | None = None,
    eval_head: str | None = None,
) -> None:
    """Parse Ultralytics' artifacts off disk into ``state.eval`` (+ ``compare``).

    ``results.csv``'s last row is Ultralytics' per-epoch **validation** metric
    (the split reserved for early-stopping/model-selection during training,
    recorded every epoch). ``val_results`` -- when the caller passed one -- is
    a fresh ``model.val(..., split='test')`` pass against the frozen holdout
    the run was never trained or tuned against. These are NOT
    interchangeable: labeling the val number as "test" overstates how the
    model will do on unseen data.

    The overall ``eval.map50`` / ``eval.map50_95`` (+ ``precision`` /
    ``recall`` / ``per_class`` when available) come from the test pass
    whenever it succeeded (``eval.split == 'test'``). When it didn't run or
    produced no usable ``box`` metrics, they fall back to the training-time
    validation numbers (``eval.split == 'val'``, no ``per_class`` -- those
    would silently be val-split numbers mislabeled as test). The val numbers
    are always additionally kept, clearly named, under ``eval.val_last`` so a
    consumer that specifically wants the training-time curve still can. The
    confusion-matrix PNG path (server filesystem; the API rewrites this to a
    servable URL before it reaches the wire) rounds out the block. When an
    incumbent is configured and reachable, the side-by-side comparison lands
    in ``state.compare``.

    ``eval_head`` (from :func:`_finalize_run`) records which detection head
    that same test-split ``val()`` pass scored -- ``"end2end"`` when the
    loaded checkpoint's NMS-free one-to-one head was explicitly forced to
    match what's actually served (see ``_finalize_run``), ``None`` for a
    model family with no such distinction. Making this explicit means a
    comparison result never silently depends on Ultralytics' own
    checkpoint-dependent ``.val()`` default.
    """
    eval_block: dict[str, Any] = {}
    row = incumbent_compare.read_results_csv_last_row(save_dir / 'results.csv')
    val_last = incumbent_compare.extract_top_level_metrics(row) if row is not None else {}

    test_summary: dict[str, float] = {}
    per_class: list[dict[str, Any]] = []
    if val_results is not None:
        test_summary = incumbent_compare.extract_test_summary(val_results)
        per_class = incumbent_compare.per_class_from_val_results(val_results)

    if test_summary and per_class:
        eval_block.update(test_summary)
        eval_block['split'] = 'test'
        eval_block['per_class'] = per_class
    else:
        eval_block.update(val_last)
        eval_block['split'] = 'val'

    if val_last:
        eval_block['val_last'] = val_last

    cm_path = save_dir / 'confusion_matrix.png'
    if cm_path.is_file():
        eval_block['confusion_matrix_path'] = str(cm_path)

    if eval_head is not None:
        eval_block['head'] = eval_head

    if eval_block:
        with state.lock:
            state.eval = eval_block

    if val_results is None or data_yaml_path is None or not per_class:
        return
    try:
        compare_block = incumbent_compare.build_compare_block(
            save_dir=save_dir,
            candidate_per_class=per_class,
            candidate_summary={k: v for k, v in eval_block.items() if k in ('map50', 'map50_95')},
            data_yaml_path=data_yaml_path,
        )
    except Exception as exc:  # a missing Triton must not tank the run
        logger.warning('compare: comparison block failed', error=str(exc))
        return
    if compare_block is not None:
        with state.lock:
            state.compare = compare_block


# ---------------------------------------------------------------------------
# Training kwargs
# ---------------------------------------------------------------------------


def _apply_reproducibility_defaults(
    train_kwargs: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[dict[str, Any], int, bool]:
    """Set ``seed=`` / ``deterministic=`` defaults on ``train_kwargs`` in place.

    Without an explicit seed Ultralytics never actually seeds its RNGs, so a
    manifest recording ``training_seed=42`` would be a fiction. ``setdefault``
    keeps an explicit caller-supplied value winning. Returns the actual values
    that will reach ``model.train()`` so the manifest records the truth.
    """
    train_kwargs.setdefault('seed', int(hyperparameters.get('seed') or 42))
    train_kwargs.setdefault('deterministic', True)
    return train_kwargs, int(train_kwargs['seed']), bool(train_kwargs['deterministic'])


def resolve_base_weights(spec: JobSpec) -> str:
    """Resolve the base weights/architecture Ultralytics should start from.

    Defaults to the family+size pretrained checkpoint (e.g. ``yolo26m.pt``).
    ``job.json``'s ``hyperparameters.model`` overrides it -- useful for a
    deployment with locally cached weights, or for a from-scratch ``.yaml``
    architecture (which needs no download).
    """
    override = spec.hyperparameters.get('model')
    if override:
        return str(override)
    return f'{spec.model_family}{spec.model_size}.pt'


def build_train_kwargs(
    spec: JobSpec,
    data_yaml_path: Path,
    local_device: str | None,
) -> tuple[dict[str, Any], int, bool]:
    """Assemble the kwargs handed to ``model.train()``."""
    train_kwargs = dict(spec.hyperparameters)
    # Trainer-only knobs Ultralytics' train() would reject as unknown args.
    train_kwargs.pop('export_batch', None)
    train_kwargs.pop('model', None)
    train_kwargs['data'] = str(data_yaml_path)
    train_kwargs.setdefault('project', RUNS_ROOT)
    train_kwargs.setdefault('name', spec.mlflow_run_name)
    # MuSGD lock-in: optimizer='auto' silently swaps it for AdamW on short runs.
    train_kwargs.setdefault('optimizer', 'MuSGD')
    train_kwargs, seed, deterministic = _apply_reproducibility_defaults(
        train_kwargs, spec.hyperparameters
    )
    if local_device is not None:
        train_kwargs.setdefault('device', local_device)
    return train_kwargs, seed, deterministic


def _resolve_multi_gpu_autobatch(
    train_kwargs: dict[str, Any], weights_name: str, n_gpus: int
) -> None:
    """Resolve ``batch < 1`` (AutoBatch) for a multi-GPU run, in place.

    Ultralytics' AutoBatch is single-GPU only ("AutoBatch with batch<1 not
    supported for Multi-GPU"). On a multi-GPU run the API's arbiter has already
    brought other GPU tenants down, so local GPU 0 is clean: measure the
    per-GPU batch there, scale to the device count (keeping a multiple of N),
    and train on all GPUs with that fixed value -- i.e. "auto" still works on
    more than one GPU.
    """
    batch = train_kwargs.get('batch')
    if not (isinstance(batch, (int, float)) and batch < 1 and n_gpus > 1):
        return
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.autobatch import autobatch

        imgsz = int(train_kwargs.get('imgsz') or DEFAULT_INPUT_SIZE)
        # 0.80 of VRAM -- more throughput than Ultralytics' cautious 0.60
        # default, still leaving headroom for activation spikes.
        # ``YOLO.model`` is annotated ``str | None`` upstream but is the loaded
        # nn.Module by the time the constructor returns.
        probe: Any = YOLO(weights_name).model
        probe = probe.to('cuda:0')
        probe.train()
        per_gpu = int(autobatch(probe, imgsz=imgsz, fraction=AUTOBATCH_VRAM_FRACTION))
        del probe
        torch.cuda.empty_cache()
        total = max(n_gpus, per_gpu * n_gpus)
        total -= total % n_gpus  # keep a multiple of the GPU count
        train_kwargs['batch'] = total
        print(f'[trainer] auto-batch: {per_gpu}/GPU x {n_gpus} GPUs -> batch={total}')
    except Exception as exc:  # probe failure falls back to a fixed batch
        fallback = OOM_FALLBACK_BATCH_PER_GPU * n_gpus
        train_kwargs['batch'] = fallback
        print(f'[trainer] auto-batch probe failed ({exc!r}); batch={fallback}')


def _is_oom_error(exc: Exception) -> bool:
    """True for memory-pressure failures a smaller batch can fix.

    Covers plain CUDA OOM plus cuBLAS/cuDNN allocation failures (OOM in
    disguise). Non-memory errors are deterministic -- retrying them just loops.
    """
    message = str(exc).lower()
    return type(exc).__name__ == 'OutOfMemoryError' or any(
        token in message
        for token in (
            'out of memory',
            'cuda oom',
            'cublas_status_alloc_failed',
            'cudnn_status_alloc_failed',
            'alloc_failed',
        )
    )


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


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------


def _build_model(
    spec: JobSpec,
    state: StatusState,
    weights_name: str,
    data_yaml_path: Path,
    total_epochs: int,
) -> Any:
    """Construct a fresh model with callbacks wired. Re-callable for OOM retry."""
    from ultralytics import YOLO

    print(f'[trainer] loading weights: {weights_name}')
    model = YOLO(weights_name)
    try:
        import mlflow_callbacks
        import yaml

        with data_yaml_path.open('r', encoding='utf-8') as fh:
            data_cfg = yaml.safe_load(fh) or {}
        mlflow_callbacks.register_callbacks(
            model,
            run_name=spec.mlflow_run_name,
            profile=spec.profile,
            seed=int(spec.hyperparameters.get('seed') or 42),
            lineage=build_lineage(spec),
            data_cfg=data_cfg,
            data_yaml_path=data_yaml_path,
        )
    except Exception as exc:  # tracking is optional
        logger.warning('mlflow callback registration failed', error=str(exc))

    cb_start, cb_end, cb_fit = _make_ultralytics_callbacks(spec, state, total_epochs)
    model.add_callback('on_train_epoch_start', cb_start)
    model.add_callback('on_train_epoch_end', cb_end)
    model.add_callback('on_fit_epoch_end', cb_fit)
    return model


def _train_with_oom_backoff(
    spec: JobSpec,
    state: StatusState,
    model: Any,
    train_kwargs: dict[str, Any],
    weights_name: str,
    data_yaml_path: Path,
    total_epochs: int,
    n_gpus: int,
) -> Any | None:
    """Run ``model.train()``, halving the batch and retrying on CUDA OOM.

    Returns the trained model (possibly a rebuilt one), or ``None`` when the
    run was cancelled -- the caller has already had ``state`` updated.
    """
    oom_attempt = 0
    while True:
        print(f'[trainer] model.train(**{json.dumps(train_kwargs, default=str)})')
        try:
            model.train(**train_kwargs)
        except CancelRequestedError:
            print('[trainer] training cancelled via .cancel sentinel')
            with state.lock:
                state.state = 'cancelled'
                state.error = 'cancelled by user'
            return None
        except KeyboardInterrupt:
            print('[trainer] training cancelled via KeyboardInterrupt')
            with state.lock:
                state.state = 'cancelled'
                state.error = 'cancelled (KeyboardInterrupt)'
            return None
        except Exception as exc:
            current = int(train_kwargs.get('batch') or 0)
            if not _is_oom_error(exc) or oom_attempt >= MAX_OOM_RETRIES or current <= n_gpus:
                raise
            oom_attempt += 1
            new_batch = max(n_gpus, current // 2)
            new_batch -= new_batch % n_gpus
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.empty_cache()
            print(
                f'[trainer] CUDA OOM at batch={current}; backing off to batch={new_batch} '
                f'(attempt {oom_attempt}/{MAX_OOM_RETRIES})'
            )
            train_kwargs['batch'] = new_batch
            # Distinct run dir so Ultralytics doesn't resume the OOM'd one.
            train_kwargs['name'] = f'{spec.mlflow_run_name}-oom{oom_attempt}'
            model = _build_model(spec, state, weights_name, data_yaml_path, total_epochs)
        else:
            return model


def _export_onnx(spec: JobSpec, model: Any, best_pt: Path) -> None:
    """Export ``best.onnx`` next to ``best.pt`` for the promote path.

    ``POST {api_prefix}/train/promote/{job_id}`` copies this ONNX into the
    Triton model repo and lets Triton's TensorRT execution accelerator JIT the
    engine (rather than shipping a pre-built plan), so ``dynamic=True`` +
    ``simplify=True`` is the right shape. ``half=False`` keeps the
    float-encoded class id precise for large class counts (Ultralytics issue
    #24428); FP16 is selected at promote time instead.

    ``batch`` caps the dynamic axis and defaults to
    :data:`DEFAULT_EXPORT_BATCH`, matching the promote config's default
    ``max_batch_size`` -- override both together via
    ``hyperparameters.export_batch``.
    """
    export_batch = int(spec.hyperparameters.get('export_batch', DEFAULT_EXPORT_BATCH))
    try:
        onnx_model = type(model)(str(best_pt))
        onnx_model.export(
            format='onnx', dynamic=True, batch=export_batch, simplify=True, half=False
        )
        print(f'[trainer] exported {best_pt.with_suffix(".onnx")} (dynamic, batch<={export_batch})')
    except Exception as exc:  # a promote can still be retried by hand
        logger.warning('onnx export failed', error=str(exc))
        with (
            contextlib.suppress(OSError),
            spec.run_log_path.open('a', encoding='utf-8') as fh,
        ):
            fh.write(f'\n[trainer] WARN onnx export failed: {exc}\n')


def _finalize_run(spec: JobSpec, state: StatusState, model: Any, data_yaml_path: Path) -> None:
    """Locate the checkpoint, export ONNX, and capture eval metrics."""
    save_dir = Path(getattr(model.trainer, 'save_dir', f'{RUNS_ROOT}/{spec.mlflow_run_name}'))
    best_pt = save_dir / 'weights' / 'best.pt'
    if not best_pt.is_file():
        logger.warning('best.pt not found', save_dir=str(save_dir))
    else:
        with state.lock:
            state.checkpoint_path = str(best_pt)
        if not copy_class_remap_to_weights_dir(spec, save_dir):
            with state.lock:
                state.class_remap_copy_failed = True
            write_status_now(spec.status_path, state)
        _export_onnx(spec, model, best_pt)

    # Fresh val pass on the test split for full per-class metrics --
    # Ultralytics writes only mAP to results.csv. Best-effort; a failure here
    # falls back to the top-level metrics.
    val_results: Any | None = None
    eval_head: str | None = None
    if best_pt.is_file():
        try:
            eval_model = type(model)(str(best_pt))
            # YOLO26 exports/serves the NMS-free one-to-one head (nms=False
            # at export -- Ultralytics forces nms=False for any end2end
            # model). `.val()` has no `end2end=` kwarg: the only real
            # toggle is the loaded model's own `.end2end` property
            # (ultralytics.nn.tasks.DetectionModel.end2end, a setter that
            # delegates to set_head_attr). Force it here so this
            # test-split re-validation scores the SAME head that's
            # actually served, rather than silently depending on whatever
            # head the reloaded checkpoint happens to default to.
            # `hasattr(..., 'one2one')` (only present on a genuine
            # dual-head build) guards against forcing end2end on a
            # non-end2end architecture, which has no one2one branch to
            # switch to and would break inference.
            if spec.model_family == 'yolo26':
                inner_model = getattr(eval_model, 'model', None)
                head_layers = (
                    getattr(inner_model, 'model', None) if inner_model is not None else None
                )
                head_module = head_layers[-1] if head_layers else None
                if (
                    inner_model is not None
                    and head_module is not None
                    and hasattr(head_module, 'one2one')
                ):
                    inner_model.end2end = True
                    eval_head = 'end2end'
            val_results = eval_model.val(
                data=str(data_yaml_path), split='test', plots=False, verbose=False
            )
        except Exception as exc:  # metrics are not worth failing a run over
            logger.warning('val pass for per-class metrics failed', error=str(exc))
            eval_head = None

    populate_eval_block(
        state, save_dir, val_results=val_results, data_yaml_path=data_yaml_path, eval_head=eval_head
    )


def run_job(spec: JobSpec) -> None:
    """Process a single job end-to-end.

    Updates ``status.json`` and ``run.log`` throughout, writes the manifest
    before cleanup (the class_remap lives under the tmp dir), and always
    removes the per-job tmp dir on exit.
    """
    spec.tmp_root.mkdir(parents=True, exist_ok=True)

    state = StatusState(
        job_id=spec.job_id,
        campaign_id=spec.campaign_id,
        state='starting',
        total_epochs=0,
    )
    state.started_at = _utcnow_iso()
    write_status_now(spec.status_path, state)

    stop_hb = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(stop_hb, state, spec.status_path), daemon=True
    )
    hb_thread.start()

    try:
        with tee_to_run_log(spec.run_log_path):
            print(f'[trainer] starting job {spec.job_id}')
            print(
                f'[trainer]   model_family={spec.model_family} size={spec.model_size}'
                f' profile={spec.profile}'
            )
            print(f'[trainer]   dataset={spec.dataset_export_dir}')
            print(
                f'[trainer]   include_classes={spec.include_classes} single_cls={spec.single_cls}'
            )

            # GPU selection. ``cuda_visible_devices`` is in HOST indices; the
            # container may see them at different local indices. We hand
            # Ultralytics the resolved local index and leave
            # CUDA_VISIBLE_DEVICES untouched -- re-exporting the host string
            # would filter to a wrong/missing local index.
            n_gpus = len([d for d in (spec.cuda_visible_devices or '').split(',') if d.strip()])
            local_device = resolve_local_device(spec.cuda_visible_devices)
            print(
                f'[trainer]   gpu_spec(host)={spec.cuda_visible_devices}'
                f' -> device(local)={local_device}'
            )

            data_yaml_path, _text_classes = prepare_dataset(spec)

            weights_name = resolve_base_weights(spec)
            total_epochs = int(spec.hyperparameters.get('epochs') or 0)
            model = _build_model(spec, state, weights_name, data_yaml_path, total_epochs)

            with state.lock:
                state.state = 'running'
                state.total_epochs = total_epochs
            write_status_now(spec.status_path, state)

            # Ultralytics' built-in MLflow integration reads these.
            os.environ.setdefault(
                'MLFLOW_TRACKING_URI', os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
            )
            os.environ['MLFLOW_RUN'] = spec.mlflow_run_name
            _guard_ultralytics_mlflow_artifact_root(
                os.environ.get('MLFLOW_EXPERIMENT_NAME', 'openprocessor')
            )

            train_kwargs, seed, deterministic = build_train_kwargs(
                spec, data_yaml_path, local_device
            )
            with state.lock:
                state.actual_seed = seed
                state.actual_deterministic = deterministic
            _resolve_multi_gpu_autobatch(train_kwargs, weights_name, n_gpus)

            trained = _train_with_oom_backoff(
                spec, state, model, train_kwargs, weights_name, data_yaml_path, total_epochs, n_gpus
            )
            if trained is None:
                return  # cancelled; state already set

            with state.lock:
                state.state = 'exporting'
            write_status_now(spec.status_path, state)

            _finalize_run(spec, state, trained, data_yaml_path)
            _capture_mlflow_run_id(spec, state)

            with state.lock:
                state.state = 'finished'
            print(f'[trainer] finished {spec.job_id}')

    except Exception as exc:  # every failure must land in status.json
        logger.exception('job failed', job_id=spec.job_id)
        tb = traceback.format_exc()
        with state.lock:
            state.state = 'failed'
            state.error = f'{type(exc).__name__}: {exc}\n{tb}'
        with (
            contextlib.suppress(OSError),
            spec.run_log_path.open('a', encoding='utf-8') as fh,
        ):
            fh.write(f'\n[trainer] FAILED: {exc}\n{tb}\n')
    finally:
        with state.lock:
            state.finished_at = _utcnow_iso()
        # Write the terminal status BEFORE stopping the heartbeat so the API
        # never sees a stale non-terminal state with a dead heartbeat.
        write_status_now(spec.status_path, state)
        stop_hb.set()
        hb_thread.join(timeout=10)
        write_status_now(spec.status_path, state)
        # Manifest before tmp cleanup -- class_remap.json lives under tmp_root,
        # and the manifest is the only copy of it that survives the rmtree
        # below other than the one in the checkpoint's weights/ dir.
        write_manifest(spec, state, class_remap=read_class_remap(spec))
        if spec.auto_quantize_bakeoff and state.state == 'finished':
            try:
                write_quant_bakeoff_job(spec, state)
            except OSError as exc:
                logger.warning('auto-quantize hook failed', job_id=spec.job_id, error=str(exc))
        if spec.tmp_root.exists():
            shutil.rmtree(spec.tmp_root, ignore_errors=True)
        # Campaign hook runs AFTER the terminal status is on disk.
        try:
            maybe_handle_campaign(spec, state)
        except Exception as exc:  # sibling bookkeeping is non-fatal
            logger.warning('campaign hook failed', job_id=spec.job_id, error=str(exc))


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


def process_one(jobs_dir: Path) -> bool:
    """Run the oldest pending job, if any. Returns True when one was handled."""
    pending = list_pending_jobs(jobs_dir)
    if not pending:
        return False
    job_path = pending[0]
    try:
        spec = parse_and_validate_job(job_path)
    except Exception as exc:  # any parse failure is a job rejection
        logger.exception('invalid job', path=str(job_path))
        reject_job(job_path, exc)
        return True
    logger.info('running job', job_id=spec.job_id)
    run_job(spec)
    return True


def watch_loop(jobs_dir: Path) -> None:
    """Poll, run, repeat. Honors SIGTERM/SIGINT for a graceful exit."""
    stop = threading.Event()

    def _signal_handler(signum: int, _frame: Any) -> None:
        logger.info('signal received, exiting watcher', signum=signum)
        stop.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    logger.info('watcher started', jobs_dir=str(jobs_dir))
    while not stop.is_set():
        try:
            handled = process_one(jobs_dir)
        except Exception:  # the watcher must outlive any single job
            logger.exception('watcher: job dispatch failed')
            handled = False
        if handled:
            continue
        for _ in range(int(POLL_INTERVAL_S * 10)):
            if stop.is_set():
                break
            time.sleep(0.1)

    logger.info('watcher stopped')


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get('OP_TRAIN_LOG_LEVEL', 'INFO'),
        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='trainer', description='Watch a jobs directory and run YOLO training jobs'
    )
    parser.add_argument(
        '--watch',
        type=Path,
        default=DEFAULT_JOBS_DIR,
        help='Directory to poll for *.job.json files',
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Process one job (or none) then exit. Used by tests.',
    )
    args = parser.parse_args(argv)

    _setup_logging()

    if args.once:
        if not process_one(args.watch):
            logger.info('--once: no pending jobs')
        return 0

    watch_loop(args.watch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

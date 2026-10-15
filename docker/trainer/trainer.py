"""Trainer watcher + training loop -- the trainer container's entrypoint.

Polls the shared jobs directory for ``*.job.json`` files and runs each through
Ultralytics: GPU selection, dataset preparation, the training call (with
cooperative cancel and CUDA-OOM batch backoff), checkpoint + ONNX export, and
metric capture. The file protocol it speaks lives in :mod:`job_protocol`,
:mod:`job_spec`, :mod:`job_status` and :mod:`job_manifest`, dataset preparation in
:mod:`dataset_prep`, cross-job campaign policy in :mod:`campaign`, and the optional comparison
against a served model in :mod:`incumbent_compare`.

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
import logging
import os
import shutil
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import trainer_kwargs
from campaign import maybe_handle_campaign, write_quant_bakeoff_job
from dataset_prep import copy_class_remap_to_weights_dir, prepare_dataset, read_class_remap
from gpu_select import HEARTBEAT_INTERVAL_S, resolve_local_device, write_trainer_capabilities
from job_manifest import write_manifest
from job_protocol import list_pending_jobs, reject_job, tee_to_run_log
from job_spec import DEFAULT_JOBS_DIR, JobSpec, _utcnow_iso, parse_and_validate_job
from job_status import StatusState, _capture_mlflow_run_id, _heartbeat_loop, write_status_now
from logutil import get_logger
from trainer_callbacks import _guard_ultralytics_mlflow_artifact_root
from trainer_eval import populate_eval_block
from trainer_fit import _build_model, _export_onnx, _train_with_oom_backoff
from trainer_kwargs import _resolve_multi_gpu_autobatch, build_train_kwargs, resolve_base_weights


logger = get_logger('trainer')


POLL_INTERVAL_S = 2.0


# ---------------------------------------------------------------------------


def _finalize_run(spec: JobSpec, state: StatusState, model: Any, data_yaml_path: Path) -> None:
    """Locate the checkpoint, export ONNX, and capture eval metrics."""
    save_dir = Path(
        getattr(model.trainer, 'save_dir', f'{trainer_kwargs.RUNS_ROOT}/{spec.mlflow_run_name}')
    )
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
            # The job's own project experiment (the project-isolation design,
            # §5.3) takes priority
            # over any env-var-only experiment name -- one trainer process
            # serves every project's jobs, so a stale process-wide env var
            # must never leak one project's runs into another's experiment.
            experiment_name = spec.mlflow_experiment or os.environ.get(
                'MLFLOW_EXPERIMENT_NAME', 'openprocessor'
            )
            os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
            _guard_ultralytics_mlflow_artifact_root(experiment_name)

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
        # After the terminal status + manifest: the API resolves the run from them.
        if spec.auto_quantize_bakeoff and state.state == 'finished':
            try:
                write_quant_bakeoff_job(spec, state)
            except Exception as exc:  # the hand-off must never fail a finished run
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
    last_heartbeat = time.monotonic()
    while not stop.is_set():
        try:
            handled = process_one(jobs_dir)
        except Exception:  # the watcher must outlive any single job
            logger.exception('watcher: job dispatch failed')
            handled = False
        now = time.monotonic()
        if now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
            write_trainer_capabilities(jobs_dir, probe_cuda=False)
            last_heartbeat = now
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
    write_trainer_capabilities(args.watch)

    if args.once:
        if not process_one(args.watch):
            logger.info('--once: no pending jobs')
        return 0

    watch_loop(args.watch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

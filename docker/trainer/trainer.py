"""The training loop of one run: Ultralytics, from GPU selection to the exported checkpoint.

:func:`run_training` is blocking and runs in a worker thread of the ``train`` handler
(:mod:`celery_tasks`): GPU selection, dataset preparation, the training call (with cooperative
cancel and CUDA-OOM batch backoff), checkpoint + ONNX export, and metric capture. Run state lives
in :mod:`job_status`, the manifest in :mod:`job_manifest`, dataset preparation in
:mod:`dataset_prep`, cross-run campaign policy in :mod:`campaign`, and the optional comparison
against a served model in :mod:`incumbent_compare`.

Nothing domain-specific is hardcoded. Everything that varies per deployment --
dataset location, class subset, hyperparameters, augmentation, which classes
are orientation-sensitive, which Triton model to compare against -- arrives
through the run's spec or the ``OP_*`` environment (see ``env.template``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trainer_kwargs
from dataset_prep import copy_class_remap_to_weights_dir, prepare_dataset
from gpu_select import resolve_local_device
from job_status import _capture_mlflow_run_id
from logutil import get_logger
from trainer_callbacks import _guard_ultralytics_mlflow_artifact_root
from trainer_eval import populate_eval_block
from trainer_fit import _build_model, _export_onnx, _train_with_oom_backoff
from trainer_kwargs import _resolve_multi_gpu_autobatch, build_train_kwargs, resolve_base_weights

from src.domain.model_families import head_kind_of


if TYPE_CHECKING:
    from job_spec import JobSpec
    from job_status import StatusState


logger = get_logger('trainer')

# What a failed Ultralytics validation pass raises; the run keeps its top-level metrics.
_EVAL_FAILURES = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    AssertionError,
    OSError,
)


# ---------------------------------------------------------------------------


def _select_served_head(eval_model: Any, model_family: str) -> str | None:
    """Make the test-split validation score the head that is served; ``None`` if it has none.

    YOLO26 exports and serves the NMS-free one-to-one head (``nms=False`` at export).
    ``.val()`` has no ``end2end=`` kwarg: the only real toggle is the loaded model's own
    ``.end2end`` property (``ultralytics.nn.tasks.DetectionModel.end2end``, a setter that
    delegates to ``set_head_attr``). Forcing it scores the SAME head that is served, rather than
    whatever head the reloaded checkpoint happens to default to. A head without the one-to-one
    branch (:func:`src.domain.model_families.head_kind_of`) keeps its own: there is no branch
    to switch to and forcing ``end2end`` would break inference.
    """
    if model_family != 'yolo26':
        return None
    inner_model = getattr(eval_model, 'model', None)
    head_layers = getattr(inner_model, 'model', None) if inner_model is not None else None
    head_module = head_layers[-1] if head_layers else None
    if inner_model is None or head_module is None or head_kind_of(head_module) != 'one2one':
        return None
    inner_model.end2end = True
    return 'end2end'


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
        _export_onnx(spec, model, best_pt)

    # Fresh val pass on the test split for full per-class metrics --
    # Ultralytics writes only mAP to results.csv. Best-effort; a failure here
    # falls back to the top-level metrics.
    val_results: Any | None = None
    eval_head: str | None = None
    if best_pt.is_file():
        try:
            eval_model = type(model)(str(best_pt))
            eval_head = _select_served_head(eval_model, spec.model_family)
            val_results = eval_model.val(
                data=str(data_yaml_path), split='test', plots=False, verbose=False
            )
        except _EVAL_FAILURES as exc:  # metrics are not worth failing a run over
            logger.warning('val pass for per-class metrics failed', error=str(exc))
            eval_head = None

    populate_eval_block(
        state, save_dir, val_results=val_results, data_yaml_path=data_yaml_path, eval_head=eval_head
    )


def run_training(spec: JobSpec, state: StatusState) -> None:
    """Train one run end to end, updating ``state`` as it goes.

    Blocking. Returns normally when the run finished or was cancelled (``state.state`` says
    which); any other failure propagates for the handler to record. The caller owns the scratch
    directory (``spec.tmp_root``) and removes it after reading what it needs from it.
    """
    spec.tmp_root.mkdir(parents=True, exist_ok=True)
    print(f'[trainer] starting run {spec.job_id}')
    print(
        f'[trainer]   model_family={spec.model_family} size={spec.model_size}'
        f' profile={spec.profile}'
    )
    print(f'[trainer]   dataset={spec.dataset_export_dir}')
    print(f'[trainer]   include_classes={spec.include_classes} single_cls={spec.single_cls}')

    # GPU selection. ``cuda_visible_devices`` is in HOST indices; the container may see them at
    # different local indices. We hand Ultralytics the resolved local index and leave
    # CUDA_VISIBLE_DEVICES untouched -- re-exporting the host string would filter to a
    # wrong/missing local index.
    n_gpus = len([d for d in (spec.cuda_visible_devices or '').split(',') if d.strip()])
    local_device = resolve_local_device(spec.cuda_visible_devices)
    print(f'[trainer]   gpu_spec(host)={spec.cuda_visible_devices} -> device(local)={local_device}')

    data_yaml_path, _text_classes = prepare_dataset(spec)

    weights_name = resolve_base_weights(spec)
    total_epochs = int(spec.hyperparameters.get('epochs') or 0)
    model = _build_model(spec, state, weights_name, data_yaml_path, total_epochs)

    with state.lock:
        state.state = 'running'
        state.total_epochs = total_epochs

    # Ultralytics' built-in MLflow integration reads these.
    os.environ.setdefault(
        'MLFLOW_TRACKING_URI', os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    )
    os.environ['MLFLOW_RUN'] = spec.mlflow_run_name
    # The run's own project experiment takes priority over any env-var-only experiment name --
    # one trainer process serves every project's runs, so a stale process-wide env var must never
    # leak one project's runs into another's experiment.
    experiment_name = spec.mlflow_experiment or os.environ.get(
        'MLFLOW_EXPERIMENT_NAME', 'openprocessor'
    )
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    _guard_ultralytics_mlflow_artifact_root(experiment_name)

    train_kwargs, seed, deterministic = build_train_kwargs(spec, data_yaml_path, local_device)
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

    _finalize_run(spec, state, trained, data_yaml_path)
    _capture_mlflow_run_id(spec, state)

    with state.lock:
        state.state = 'finished'
    print(f'[trainer] finished {spec.job_id}')

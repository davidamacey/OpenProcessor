"""Model construction, the OOM-backoff training call and ONNX export."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

from job_manifest import build_lineage
from job_spec import DEFAULT_EXPORT_BATCH, JobSpec
from logutil import get_logger
from trainer_callbacks import CancelRequestedError, _make_ultralytics_callbacks
from trainer_kwargs import MAX_OOM_RETRIES, _is_oom_error


if TYPE_CHECKING:
    from pathlib import Path

    from job_status import StatusState


logger = get_logger('trainer.fit')


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

"""Ultralytics ``train()`` kwargs, reproducibility defaults, multi-GPU AutoBatch and OOM classification."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from job_spec import DEFAULT_INPUT_SIZE, JobSpec
from logutil import get_logger


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger('trainer.trainer_kwargs')


# Ultralytics' ``project=`` dir for run outputs. This MUST be the same absolute
# path in every container that later reads a path recorded in
# status.json/manifest.json (``checkpoint_path``, the ONNX sibling) -- the API
# container resolves ``checkpoint_path`` verbatim when promoting, so a
# trainer-private mount alias 404s the moment promote runs. Both containers
# mount the same host dir at this path; see docker-compose.yml.
RUNS_ROOT = os.environ.get('OP_TRAIN_RUNS_ROOT', '/var/lib/openprocessor/training_runs')


# CUDA-OOM backoff: halve the batch and retry with a clean model, so an
# unattended run completes instead of dying on a too-large batch.
MAX_OOM_RETRIES = 5
AUTOBATCH_VRAM_FRACTION = 0.80
OOM_FALLBACK_BATCH_PER_GPU = 16


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

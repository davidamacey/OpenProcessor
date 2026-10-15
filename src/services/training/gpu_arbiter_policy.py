"""GPU arbiter policy: which containers a claim stops, and the state files.

Split out of ``gpu_arbiter.py``: the result and error types, the
``CUDA_VISIBLE_DEVICES`` parsing, the scope rules deciding which configured
containers a claim must stop, the training-lock file and the worker pause
sentinel. Nothing here talks to docker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config, get_gpu_arbiter_config
from src.core.logging import get_logger
from src.services.training.pause_sentinel import arbiter_body, arbiter_may_clear


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


def _state_dir() -> Path:
    return get_curation_config().state_dir


def _default_sentinel_path() -> Path:
    # Suffix must match CurationConfig.pause_sentinel_path (readers).
    return _state_dir() / 'vlm_worker' / 'pause.sentinel'


def _default_lock_path() -> Path:
    return _state_dir() / 'training_gpus.lock'


@dataclass(frozen=True)
class ArbiterAction:
    """Returned by every arbiter call so the caller can log what happened."""

    action: str  # 'sentinel_set' | 'sentinel_cleared' | 'gpu_services_stopped' | 'gpu_services_started' | 'noop'
    detail: str = ''
    # Project owning the active run (projects_plan.md §5.3); None when
    # nothing is active or it can't be attributed (e.g. the lock closing
    # the claim->write race before job.json's project is known).
    holder_project: str | None = None


class GpuArbiterStopFailedError(RuntimeError):
    """A claim needed to stop a configured GPU-resident container and
    couldn't (docker SDK/socket unavailable, or the stop call failed).
    Used to fall back to a sentinel-only pause instead, which doesn't
    stop the container -- training could start next to it on the same
    GPU. Callers (``/train/start``, ``/train/start_campaign``) must
    catch this and refuse with 409 rather than proceed.
    """


def parse_cuda_visible_devices(spec_value: str | None) -> list[int]:
    """Parse a CUDA_VISIBLE_DEVICES string into the integer device list.

    >>> parse_cuda_visible_devices('0,2')
    [0, 2]
    >>> parse_cuda_visible_devices(None)
    []
    >>> parse_cuda_visible_devices('  0 ,  1 ')
    [0, 1]
    """
    if not spec_value:
        return []
    out: list[int] = []
    for raw in spec_value.split(','):
        tok = raw.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            logger.warning('cuda_visible_devices_unparseable', token=tok)
    return out


def containers_to_stop(cuda_visible_devices: str | None) -> tuple[str, ...]:
    """Ordered names of configured containers this claim must stop.

    A *scoped* container (``name@ids`` in ``OP_GPU_ARBITER_CONTAINERS``,
    see :class:`src.config.gpu_arbiter.GpuArbiterConfig`) is stopped
    whenever the claim intersects its GPU set, regardless of claim size --
    a lone-GPU claim on a GPU that hosts a large service must stop that
    service, not just pause a paired worker. An *unscoped* container keeps
    the original behavior: stopped only when the claim spans more than one
    GPU. Order follows ``GpuArbiterConfig.container_gpus`` (== the order
    containers were configured in).
    """
    claim = frozenset(parse_cuda_visible_devices(cuda_visible_devices))
    multi_gpu_claim = len(claim) > 1
    names: list[str] = []
    for name, scope in get_gpu_arbiter_config().container_gpus:
        if scope is None:
            if multi_gpu_claim:
                names.append(name)
        elif claim & scope:
            names.append(name)
    return tuple(names)


def needs_service_stop(cuda_visible_devices: str | None) -> bool:
    """``True`` iff this claim requires stopping at least one configured container."""
    return bool(containers_to_stop(cuda_visible_devices))


def sentinel_path(custom: Path | None = None) -> Path:
    return custom or _default_sentinel_path()


def lock_path(custom: Path | None = None) -> Path:
    return custom or _default_lock_path()


def set_training_lock(
    cuda_visible_devices: str | None,
    *,
    path: Path | None = None,
) -> None:
    """Record that a training run owns the configured GPU-resident services."""
    target = lock_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({'claimed_at': time.time(), 'cuda_visible_devices': cuda_visible_devices or ''}),
        encoding='utf-8',
    )
    logger.info('arbiter_training_lock_set', path=str(target), devices=cuda_visible_devices)


def clear_training_lock(*, path: Path | None = None) -> None:
    """Remove the training lock. Idempotent."""
    target = lock_path(path)
    try:
        target.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning('arbiter_training_lock_unlink_failed', error=str(exc))
        return
    logger.info('arbiter_training_lock_cleared', path=str(target))


def read_training_lock(*, path: Path | None = None) -> dict[str, Any] | None:
    """Return the lock payload, or ``None`` if absent/unreadable."""
    target = lock_path(path)
    try:
        typed: dict[str, Any] | None = json.loads(target.read_text(encoding='utf-8'))
        return typed
    except (FileNotFoundError, ValueError, OSError):
        return None


async def pause_gpu_worker(
    *,
    sentinel: Path | None = None,
) -> ArbiterAction:
    """Touch the pause sentinel so a paired CPU-side worker idles.

    The GPU-resident model server(s) are left alone -- only the polling
    worker backs off. This is the single-GPU regime: trainer and the
    other configured service share the host's CPU/RAM but live on
    different GPUs.
    """
    target = sentinel_path(sentinel)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return ArbiterAction(action='noop', detail=f'sentinel already at {target}')
    target.write_text(arbiter_body(), encoding='utf-8')
    logger.info('arbiter_sentinel_set', path=str(target))
    return ArbiterAction(action='sentinel_set', detail=str(target))


async def resume_gpu_worker(
    *,
    sentinel: Path | None = None,
) -> ArbiterAction:
    """Remove the arbiter's pause sentinel (never another owner's, unless stale). Idempotent."""
    target = sentinel_path(sentinel)
    if not target.exists():
        return ArbiterAction(action='noop', detail=f'no sentinel at {target}')
    if not arbiter_may_clear(target):
        # e.g. `openprocessor vlm use` paused the workers for a model swap (#127)
        return ArbiterAction(action='noop', detail=f"sentinel at {target} is not the arbiter's")
    try:
        target.unlink()
    except OSError as exc:
        logger.warning('arbiter_sentinel_unlink_failed', error=str(exc))
        return ArbiterAction(action='noop', detail=f'unlink failed: {exc}')
    logger.info('arbiter_sentinel_cleared', path=str(target))
    return ArbiterAction(action='sentinel_cleared', detail=str(target))

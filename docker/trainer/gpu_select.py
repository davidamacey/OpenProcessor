"""GPU attachment: host-to-local device mapping and the trainer capabilities file the API preflight reads."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from job_spec import _utcnow_iso
from job_status import _atomic_write_json
from logutil import get_logger


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger('trainer.gpu_select')


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


# Sibling of the job-queue dir; the API's preflight reads this to reject a
# training request for a host GPU this trainer container isn't actually
# attached to (TR-2 GPU scoping fix) instead of accepting it and hanging in
# `starting` forever.
TRAINER_CAPABILITIES_FILENAME = '.trainer_capabilities.json'


def _probe_cuda_device_count() -> int | None:
    """``torch.cuda.device_count()``, run in an isolated, short-lived
    subprocess rather than in this process.

    this trainer runs as a long-lived daemon (``--watch`` never
    returns) sharing one process for its entire container lifetime.
    ``write_trainer_capabilities`` used to ``import torch`` and call this
    in-process, at startup, exactly once -- but that is also the *first*
    CUDA touch this process ever makes, and on the torch/CUDA-driver
    combination this image resolves at build time (``torch>=2.6,<3``,
    unpinned to a minor), that can be enough to lazily initialize a CUDA
    context that then lives for the rest of the container's uptime. This
    matches the observed symptom exactly: ~1.1 GB held on an otherwise
    completely idle curation-trainer between training runs, with nothing
    else in this module touching CUDA outside of an actual job run.

    Paying for the probe in a subprocess means whatever context it
    initializes (if any) is released the instant that subprocess exits --
    the long-lived watcher process itself never calls into CUDA just to
    answer "how many GPUs do I see". A real training job still imports
    torch and initializes CUDA in *this* process, same as before, because
    a job holding VRAM for its own duration is expected; only the idle
    startup probe changes.
    """
    import subprocess  # nosec B404 - only used with a fixed argv, see below
    import sys

    try:
        result = subprocess.run(  # nosec B603 - fixed argv (sys.executable), no user input
            [sys.executable, '-c', 'import torch; print(torch.cuda.device_count())'],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception:
        logger.warning('trainer_capabilities_cuda_probe_failed', exc_info=True)
        return None


# How often watch_loop re-touches .trainer_capabilities.json's written_at
# while idle. The API's probe_trainer_reachable (src/services/training/
# gpu_arbiter.py TRAINER_HEARTBEAT_STALE_SECONDS=120) treats anything
# older than 4x this as stale -- this is the socket-free signal that
# replaced the previous docker-only reachability check.
HEARTBEAT_INTERVAL_S = 30.0

# Cached across heartbeats so re-touching written_at doesn't re-fork a
# subprocess to re-probe CUDA every 30s -- only the first write (startup)
# and any explicit re-probe pay that cost.
_capabilities_cache: dict[str, Any] = {}


def write_trainer_capabilities(jobs_dir: Path, *, probe_cuda: bool = True) -> None:
    """Publish this container's GPU attachment for the API's preflight check.

    Writes ``<jobs_dir>/.trainer_capabilities.json`` atomically:
    ``gpu_order`` (:func:`_gpu_order`, empty = unrestricted -- every GPU at
    its host index), ``visible_count`` (:func:`_probe_cuda_device_count`),
    ``build_sha`` (this image's baked ``OP_BUILD_SHA``), and ``written_at``.

    Called once at startup (``probe_cuda=True``) and then again every
    ``HEARTBEAT_INTERVAL_S`` from :func:`watch_loop` while idle
    (``probe_cuda=False``, reusing the cached ``visible_count``) purely to
    refresh ``written_at`` -- the API's reachability probe reads that
    timestamp's freshness instead of needing a docker socket.

    Best-effort: any failure (no CUDA, no write permission) is logged and
    swallowed -- an older trainer image or a probe failure must only
    downgrade the API's check to a warning, never crash trainer startup.
    """
    if probe_cuda or 'visible_count' not in _capabilities_cache:
        _capabilities_cache['visible_count'] = _probe_cuda_device_count()
    payload = {
        'gpu_order': list(_gpu_order()),
        'visible_count': _capabilities_cache['visible_count'],
        'build_sha': os.environ.get('OP_BUILD_SHA'),
        'written_at': _utcnow_iso(),
    }
    try:
        jobs_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(jobs_dir / TRAINER_CAPABILITIES_FILENAME, payload)
    except OSError:
        logger.warning('trainer_capabilities_write_failed', exc_info=True)
    else:
        logger.info('trainer_capabilities_written', gpu_order=list(_gpu_order()))


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

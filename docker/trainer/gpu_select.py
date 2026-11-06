"""GPU attachment: the host-to-local device mapping of ``OP_TRAIN_GPU_ORDER``.

The cards the worker sees are recorded by the job runtime at worker start (``gpu_devices`` and
``worker_capabilities``); the API preflight reads them there.
"""

from __future__ import annotations

import os

from src.domain.gpu_order import parse_order


# Host GPU ids this container is attached to, in the exact order Docker's
# ``deploy.resources.reservations.devices[].device_ids`` lists them.
# Container-local CUDA indices are positional within that list, NOT the host
# ids -- so a container given ``device_ids: ['0', '2']`` sees host GPU 2 as
# local index 1. The run spec's ``cuda_visible_devices`` names HOST ids (the
# API's GPU arbiter reasons in host ids), so they must be translated.
#
# Unset (the default) means the container sees every GPU at its host index and
# the job's device string is passed through unchanged. Set it to the compose
# file's ``device_ids`` list whenever the trainer is pinned to a GPU subset;
# they MUST be kept in sync.
def _gpu_order() -> tuple[int, ...]:
    return parse_order(os.environ.get('OP_TRAIN_GPU_ORDER'))


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

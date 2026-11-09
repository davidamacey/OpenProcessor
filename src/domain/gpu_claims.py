"""Which GPU-resident services a GPU claim has to stop (pure policy, no Docker, no files).

A claim is a run (train, probe, bake-off) that holds cards. The deployment names the services
that live on those cards in ``OP_GPU_ARBITER_CONTAINERS`` (Compose service names, optionally
``service@gpu-ids``; :class:`src.config.gpu_arbiter.GpuArbiterConfig`). The arbiter tick decides
from these rules; the preflight shows them to the person starting a run.
"""

from __future__ import annotations

import logging

from src.config import get_gpu_arbiter_config


logger = logging.getLogger(__name__)

BAKEOFF = 'bakeoff'


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
            logger.warning('cuda_visible_devices_unparseable token=%s', tok)
    return out


def containers_to_stop(cuda_visible_devices: str | None) -> tuple[str, ...]:
    """Ordered names of configured services this claim must stop.

    A *scoped* service (``name@ids``) is stopped whenever the claim intersects its GPU set,
    regardless of claim size: a lone-GPU claim on a GPU that hosts a large service must stop that
    service. An *unscoped* service is stopped only when the claim spans more than one GPU. Order
    follows ``GpuArbiterConfig.container_gpus``.
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
    """``True`` iff this claim requires stopping at least one configured service."""
    return bool(containers_to_stop(cuda_visible_devices))


def claim_services(kind: str, cuda_visible_devices: str | None) -> tuple[str, ...]:
    """The services a held claim of ``kind`` keeps stopped.

    A bake-off names no cards of its own: it claims every configured service unless
    ``OP_BAKEOFF_HOST_GPUS`` scopes it to the evaluator's host cards (then only an intersecting
    service stays stopped). A train or probe claim goes by its own ``cuda_visible_devices``; an
    unknown device list (``None``) claims every configured service, so a claim never stops less
    than it might need.
    """
    config = get_gpu_arbiter_config()
    if kind == BAKEOFF:
        scope = config.bakeoff_host_gpus
        return containers_to_stop(scope) if scope else config.containers
    if cuda_visible_devices is None:
        return config.containers
    return containers_to_stop(cuda_visible_devices)

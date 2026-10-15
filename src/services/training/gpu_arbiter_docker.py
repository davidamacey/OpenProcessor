"""GPU arbiter container control over the docker SDK.

Split out of ``gpu_arbiter.py``: the docker client probe, the sync stop,
start and status helpers, and ``stop_gpu_services`` /
``start_gpu_services``. Stopping fails closed with
:class:`GpuArbiterStopFailedError`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.config import get_gpu_arbiter_config
from src.core.logging import get_logger
from src.services.training.gpu_arbiter_policy import (
    ArbiterAction,
    GpuArbiterStopFailedError,
    pause_gpu_worker,
    resume_gpu_worker,
)


logger = get_logger(__name__)
_docker_unavailable_logged = [False]  # G-15: warn once per outage, not per call
_DOCKER_STOP_TIMEOUT = 30


def _docker_client() -> Any:
    """Return a docker SDK client over the mounted socket, or ``None`` (warns once
    per outage, not per call -- G-15). Imported lazily so the module stays
    importable without the ``docker`` package or socket (unit tests)."""
    try:
        import docker

        client = docker.from_env()  # type: ignore[attr-defined]
        client.ping()
        _docker_unavailable_logged[0] = False
        return client
    except Exception as exc:
        if not _docker_unavailable_logged[0]:  # G-15: warn once per outage
            logger.warning('arbiter_docker_unavailable', error=str(exc))
            _docker_unavailable_logged[0] = True
        return None


def docker_client_available() -> bool:
    """Preflight probe -- can a docker client reach the daemon?"""
    return _docker_client() is not None


def _stop_containers_sync(client: Any, names: tuple[str, ...]) -> list[str]:
    """Stop each named container if running. Returns the names stopped."""
    import docker.errors

    stopped: list[str] = []
    for name in names:
        try:
            container = client.containers.get(name)
        except docker.errors.NotFound:
            continue
        if container.status == 'running':
            container.stop(timeout=_DOCKER_STOP_TIMEOUT)
            stopped.append(name)
    return stopped


def _start_containers_sync(client: Any, names: tuple[str, ...]) -> list[str]:
    """Start each named container if not already running. Returns names started."""
    import docker.errors

    started: list[str] = []
    for name in names:
        try:
            container = client.containers.get(name)
        except docker.errors.NotFound:
            continue
        container.reload()
        if container.status != 'running':
            container.start()
            started.append(name)
    return started


def _container_status_sync(client: Any, name: str) -> str | None:
    """Return the named container's status, or ``None`` if it doesn't exist."""
    import docker.errors

    try:
        container = client.containers.get(name)
    except docker.errors.NotFound:
        return None
    container.reload()
    typed: str | None = container.status
    return typed


async def stop_gpu_services(
    *,
    containers: tuple[str, ...] | None = None,
) -> ArbiterAction:
    """Stop the configured GPU-resident containers to free every configured GPU.

    ``containers`` defaults to ``GpuArbiterConfig.containers``. Empty
    (generic-install default) = pure no-op, no docker SDK touched.

    Otherwise uses the docker SDK over the mounted socket; restart later
    preserves each container's original config (GPU pins included).

    Fails closed -- raises :class:`GpuArbiterStopFailedError` if the
    docker SDK/socket is unavailable or the stop call fails, rather than
    falling back to a sentinel (which pauses a paired *worker*, not a
    sibling container sharing the GPU). Callers must not proceed (or
    write job.json) when this raises.
    """
    names = containers if containers is not None else get_gpu_arbiter_config().containers
    if not names:
        return ArbiterAction(action='noop', detail='no GPU-resident containers configured')
    client = _docker_client()
    if client is None:
        raise GpuArbiterStopFailedError(
            f'docker SDK/socket unavailable in the API container -- cannot stop {names!r}'
        )
    try:
        stopped = await asyncio.to_thread(_stop_containers_sync, client, names)
    except Exception as exc:
        logger.warning('arbiter_gpu_stop_failed', error=str(exc))
        raise GpuArbiterStopFailedError(f'stop failed for {names!r}: {exc}') from exc
    # Belt-and-suspenders: also set the sentinel so a worker that somehow
    # comes back up mid-run still pauses.
    await pause_gpu_worker()
    logger.info('arbiter_gpu_services_stopped', stopped=stopped)
    return ArbiterAction(
        action='gpu_services_stopped',
        detail=','.join(stopped) if stopped else 'none were running',
    )


async def start_gpu_services(
    *,
    containers: tuple[str, ...] | None = None,
) -> ArbiterAction:
    """Restart the configured GPU-resident containers and clear the pause sentinel."""
    names = containers if containers is not None else get_gpu_arbiter_config().containers
    if not names:
        return ArbiterAction(action='noop', detail='no GPU-resident containers configured')
    client = _docker_client()
    if client is None:
        await resume_gpu_worker()
        return ArbiterAction(
            action='sentinel_cleared',
            detail='docker SDK unavailable; cleared sentinel only',
        )
    try:
        started = await asyncio.to_thread(_start_containers_sync, client, names)
    except Exception as exc:
        logger.warning('arbiter_gpu_start_failed', error=str(exc))
        return ArbiterAction(action='noop', detail=f'start failed: {exc}')
    await resume_gpu_worker()
    logger.info('arbiter_gpu_services_started', started=started)
    return ArbiterAction(
        action='gpu_services_started',
        detail=','.join(started) if started else 'all already running',
    )

"""Trainer container reachability probe used by ``/train/preflight``.

Split out of :mod:`src.services.training.gpu_arbiter` (the 700-LOC ratchet)
-- this stays a satellite of that module: it borrows ``_docker_client``/
``_container_status_sync`` from there for the optional docker-confirmation
path, but its primary signal (the trainer's heartbeat file) never touches
docker at all.

Without this probe, submitting a job writes ``job.json`` and sits in
``queued`` forever with no error if the trainer was never started;
``/train/preflight``/``/train/start`` call :func:`probe_trainer_reachable`
to surface that instead.

F-72 regression this module fixes: the original implementation only had a
docker-socket-based check, which reported ``block`` whenever the API
container had no socket mounted -- the *stock* deployment (no
``docker-compose.gpu-arbiter.yml`` overlay) -- even with a perfectly
healthy trainer, so preflight/`/start` 422'd unconditionally on every
stock install. The primary signal is now the trainer's own heartbeat file
(:data:`TRAINER_CAPABILITIES_FILENAME`, refreshed every ~30s by the
trainer's watch loop, see ``docker/trainer/trainer.py``
``HEARTBEAT_INTERVAL_S``), which needs no docker socket. "Can't tell"
(missing/stale heartbeat, no docker socket either) is now at most
``warn``, never ``block``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from src.config import get_gpu_arbiter_config
from src.services.training.gpu_arbiter import _container_status_sync, _docker_client


# Filename the trainer's watch loop refreshes every HEARTBEAT_INTERVAL_S
# (docker/trainer/trainer.py) via write_trainer_capabilities(), next to
# job.json in the shared /jobs volume. Duplicated (not imported) from
# src/routers/curation_train.py's own TRAINER_CAPABILITIES_FILENAME and
# the trainer image's copy -- same reasoning as those two: the trainer
# ships as its own image with no src/ dependency, so there is nothing to
# share an import with. tests/curation/test_trainer_protocol.py
# conformance-tests all copies against the literal name.
TRAINER_CAPABILITIES_FILENAME = '.trainer_capabilities.json'

# The trainer's watch loop rewrites this file every ~30s while idle
# (docker/trainer/trainer.py HEARTBEAT_INTERVAL_S). A generous multiple of
# that cadence tells "trainer alive but a slow poll tick" apart from
# "trainer container is actually gone" without needing the docker socket
# at all.
TRAINER_HEARTBEAT_STALE_SECONDS = 120.0


def _read_trainer_capabilities_file() -> dict[str, Any] | None:
    from src.services.training.jobs import _resolve_jobs_dir

    path = _resolve_jobs_dir() / TRAINER_CAPABILITIES_FILENAME
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _trainer_heartbeat_age_seconds(caps: dict[str, Any]) -> float | None:
    written_at = caps.get('written_at')
    if not isinstance(written_at, str):
        return None
    try:
        written = datetime.fromisoformat(written_at)
    except ValueError:
        return None
    if written.tzinfo is None:
        written = written.replace(tzinfo=UTC)
    return (datetime.now(UTC) - written).total_seconds()


async def probe_trainer_reachable(
    *,
    container_name: str | None = None,
) -> tuple[str, str]:
    """Check whether the configured trainer container is up and reachable.

    ``container_name`` defaults to ``GpuArbiterConfig.trainer_container``.
    Unset (generic-install default, no sibling container to probe) ->
    ``('ok', ...)`` rather than treating "not configured" as a failure.

    Returns ``(severity, detail)`` with ``severity`` one of ``'ok'``,
    ``'warn'``, ``'block'``.

    The **primary** signal is the trainer's heartbeat file
    (:data:`TRAINER_CAPABILITIES_FILENAME`, refreshed every ~30s by the
    trainer's watch loop) -- reading a file on the shared ``/jobs`` volume
    needs no docker socket, so this works on a stock install with no
    ``docker-compose.gpu-arbiter.yml`` overlay. A fresh heartbeat is
    ``'ok'``.

    A missing or stale heartbeat (older trainer image, container
    mid-restart, or genuinely down) is at most ``'warn'`` -- "can't tell"
    must never block ``/train/start``.

    The docker SDK/socket path (only present when the gpu-arbiter overlay
    mounts it) is used purely as an **optional, confirming** secondary
    check, and only consulted once the heartbeat itself is missing/stale:
    if docker is reachable and confirms the container is absent/stopped,
    that upgrades the warn to a definitive ``'block'``; if docker can't
    see it either (or isn't mounted), it stays ``'warn'``.
    """
    name = (
        container_name if container_name is not None else get_gpu_arbiter_config().trainer_container
    )
    if not name:
        return 'ok', 'no trainer container configured -- reachability probe not applicable'

    caps = _read_trainer_capabilities_file()
    age = _trainer_heartbeat_age_seconds(caps) if caps is not None else None
    if age is not None and age <= TRAINER_HEARTBEAT_STALE_SECONDS:
        return 'ok', f'{name!r} heartbeat fresh ({age:.0f}s ago)'

    if age is not None:
        heartbeat_detail = f'heartbeat stale ({age:.0f}s ago)'
    elif caps is not None:
        heartbeat_detail = 'heartbeat file present but unparseable'
    else:
        heartbeat_detail = (
            f'no heartbeat file yet ({TRAINER_CAPABILITIES_FILENAME}) -- '
            'trainer container may not have started'
        )

    client = _docker_client()
    if client is None:
        return 'warn', f'{name!r}: {heartbeat_detail}; cannot verify further (no docker socket)'
    try:
        import asyncio

        state = await asyncio.to_thread(_container_status_sync, client, name)
    except Exception as exc:
        return 'warn', f'{name!r}: {heartbeat_detail}; docker probe also failed: {exc}'
    if state is None:
        return 'block', f'{name!r} container does not exist ({heartbeat_detail})'
    if state != 'running':
        return (
            'block',
            f'{name!r} container status={state!r}, expected running ({heartbeat_detail})',
        )
    return 'ok', f'{name!r} is running (docker-confirmed; {heartbeat_detail})'


__all__ = [
    'TRAINER_CAPABILITIES_FILENAME',
    'TRAINER_HEARTBEAT_STALE_SECONDS',
    'probe_trainer_reachable',
]

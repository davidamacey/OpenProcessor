"""``op.periodic.gpu_arbiter``: stop the GPU-resident services a run needs, start them after.

Scheduled only when ``OP_GPU_ARBITER_CONTAINERS`` is set and consumed only by
``gpu-arbiter-worker`` (queue ``arbiter``), the one process that mounts the Docker socket. A
train, probe or bake-off handler reserves its cards (``gpu_reservations``) and then waits until
this tick has stopped the services its claim needs and recorded it. The record is the
``arbiter`` section of ``deployment_settings('vlm_paused_for_training')``
(:mod:`src.domain.gpu_arbiter_state`); the same document pauses the VLM workers.

Each pass is two transactions around the Docker calls, under the ``GPU_ARBITER`` advisory lock:

1. intent: delete reservations whose holder is no longer running, read the claims, and write
   them to the record (the acknowledgements of claims that ended are dropped);
2. the Docker calls, in a worker thread: stop what the claims need, then start what this arbiter
   stopped earlier and no claim needs now. A service somebody else stopped is never started;
3. outcome: write what is stopped, which claims are confirmed (every service it needs is down)
   and any Docker error.

A claim is acknowledged only once its services are down: a Docker failure acknowledges nothing it
did not confirm, and the handler fails closed (``gpu_busy``) instead of starting next to a service
that is still on the card. With no claims, nothing stopped and no error the record is removed and
the tick writes nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from src.config import get_gpu_arbiter_config
from src.config.advisory_locks import GPU_ARBITER
from src.core.logging import get_logger
from src.db.repos import gpu_arbiter as repo
from src.db.session import async_session_factory, deployment_tx
from src.domain import gpu_arbiter_state as state
from src.domain.gpu_claims import claim_services
from src.services.periodic import gpu_arbiter_docker as docker_control
from src.services.periodic.runner import tick_lock


logger = get_logger(__name__)


@dataclass(frozen=True)
class _Intent:
    claims: tuple[repo.Claim, ...]
    previous: state.ArbiterRecord


@dataclass(frozen=True)
class _Applied:
    stopped: tuple[str, ...]
    down: frozenset[str]
    error: str | None


def _read_intent(session: Any) -> _Intent:
    repo.release_dead_holders(session)
    claims = tuple(repo.active_claims(session))
    previous = state.arbiter_record(repo.read_value(session))
    ids = {c.job_id for c in claims}
    now_claims = tuple(sorted(ids))
    acknowledged = tuple(a for a in previous.acknowledged if a in ids)
    if (now_claims, acknowledged) != (previous.claims, previous.acknowledged):
        repo.set_section(
            session,
            state.ARBITER,
            state.ArbiterRecord(
                claims=now_claims,
                stopped=previous.stopped,
                acknowledged=acknowledged,
                error=previous.error,
            ).to_json(),
        )
    return _Intent(claims, previous)


def _apply(wanted: tuple[str, ...], previous: state.ArbiterRecord) -> _Applied:
    """Stop the wanted services, then start the ones this arbiter stopped and nobody wants."""
    kept = [s for s in previous.stopped if s in wanted]
    release = tuple(s for s in previous.stopped if s not in wanted)
    if not wanted and not release:
        return _Applied((), frozenset(), None)
    try:
        client = docker_control.docker_client()
        project = docker_control.compose_project(client)
    except docker_control.ArbiterDockerError as exc:
        return _Applied(previous.stopped, frozenset(), str(exc))
    stopped = list(kept)
    down: set[str] = set()
    errors: list[str] = []
    for service in wanted:  # one at a time: a failure keeps what was confirmed before it
        try:
            if docker_control.stop_services(client, project, (service,)):
                stopped.append(service)
            down.add(service)
        except docker_control.DOCKER_ERRORS as exc:
            errors.append(f'stop {service}: {exc}')
    for service in release:
        try:
            docker_control.start_services(client, project, (service,))
        except docker_control.DOCKER_ERRORS as exc:
            stopped.append(service)  # still ours to start on a later pass
            errors.append(f'start {service}: {exc}')
    ordered = tuple(s for s in get_gpu_arbiter_config().containers if s in stopped)
    extra = tuple(s for s in stopped if s not in ordered)
    return _Applied(ordered + extra, frozenset(down), '; '.join(errors) or None)


def _record_outcome(session: Any, intent: _Intent, applied: _Applied) -> state.ArbiterRecord:
    confirmed = tuple(
        sorted(
            c.job_id
            for c in intent.claims
            if set(claim_services(c.kind, c.cuda_visible_devices)) <= applied.down
        )
    )
    record = state.ArbiterRecord(
        claims=tuple(sorted(c.job_id for c in intent.claims)),
        stopped=applied.stopped,
        acknowledged=confirmed,
        error=applied.error,
    )
    if not record.claims and not record.stopped and record.error is None:
        repo.drop_section(session, state.ARBITER)
    else:
        repo.set_section(session, state.ARBITER, record.to_json())
    return record


def _wanted(intent: _Intent) -> tuple[str, ...]:
    needed = {s for c in intent.claims for s in claim_services(c.kind, c.cuda_visible_devices)}
    return tuple(s for s in get_gpu_arbiter_config().containers if s in needed)


async def tick() -> dict[str, Any]:
    """One arbiter pass; returns what it did for the task's log line."""
    if not get_gpu_arbiter_config().containers:
        return {'skipped': 'no GPU-resident services configured'}
    async with tick_lock(GPU_ARBITER, 0) as held:
        if not held:
            return {'skipped': 'busy'}
        async with (
            async_session_factory()() as session,
            deployment_tx(session, 'gpu arbiter intent'),
        ):
            intent = await session.run_sync(_read_intent)
        applied = await asyncio.to_thread(_apply, _wanted(intent), intent.previous)
        async with (
            async_session_factory()() as session,
            deployment_tx(session, 'gpu arbiter outcome'),
        ):
            record = await session.run_sync(lambda s: _record_outcome(s, intent, applied))
    if record.error:
        logger.warning('gpu_arbiter_docker_failed', error=record.error, claims=len(record.claims))
    return {
        'claims': len(record.claims),
        'stopped': list(record.stopped),
        'acknowledged': len(record.acknowledged),
        'error': record.error,
    }

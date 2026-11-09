"""Is a trainer there? A live ``gpu_train`` consumer and the capabilities its worker recorded.

``/train/preflight`` and ``/train/start`` ask this before queueing, so a run is not left
``queued`` for days when no trainer was ever started. The consumer is found with
``inspect().active_queues()`` (Celery docs, "Inspecting workers",
https://docs.celeryq.dev/en/stable/userguide/workers.html#inspecting-workers, timeout 1 s). The
capabilities are the ``worker_capabilities`` row the worker wrote at start (the cards it sees, as
UUIDs) joined with ``gpu_devices`` for the host card numbers the API reasons in.

A worker row has no heartbeat while the worker idles (the pool child that wrote it exits after
each run), so "fresh" means the row belongs to a host that has a live consumer now, not that its
timestamp is young.
"""

from __future__ import annotations

import asyncio
from typing import Any

import redis
from kombu.exceptions import KombuError
from sqlalchemy import select

from src.clients import task_queue
from src.config.env import env_str
from src.config.queue import TaskQueues
from src.db.models.jobs import GpuDevice, WorkerCapability
from src.db.session import async_session_factory, deployment_tx
from src.domain.gpu_order import host_id_at, parse_order


INSPECT_TIMEOUT_S = 1.0
BROKER_ERRORS = (task_queue.QueueUnavailable, KombuError, redis.RedisError, OSError)


def _inspect_active_queues() -> dict[str, list[dict[str, Any]]] | None:
    app = task_queue.producer_app()
    return app.control.inspect(timeout=INSPECT_TIMEOUT_S).active_queues()  # type: ignore[no-any-return]


async def consumer_hosts(queue: str = TaskQueues.GPU_TRAIN) -> set[str]:
    """Host names of the workers consuming ``queue``; raises on a broker failure."""
    replies = await asyncio.to_thread(_inspect_active_queues)
    hosts: set[str] = set()
    for node, queues in (replies or {}).items():
        if any(q.get('name') == queue for q in queues):
            hosts.add(node.partition('@')[2] or node)
    return hosts


async def _capabilities(queue: str) -> list[WorkerCapability]:
    async with async_session_factory()() as session, deployment_tx(session, 'trainer capabilities'):
        found = await session.execute(
            select(WorkerCapability).where(WorkerCapability.queue == queue)
        )
        return list(found.scalars())


async def read_trainer_gpu_order(queue: str = TaskQueues.GPU_TRAIN) -> list[int] | None:
    """Host card numbers the live trainer sees, or ``None`` when unknown (no live consumer, no
    capabilities row for its host, or the broker cannot be asked).

    A worker records the cards as it numbers them inside its container; ``OP_TRAIN_GPU_ORDER``
    (the same value the trainer container gets) maps those positions to host numbers.
    """
    try:
        hosts = await consumer_hosts(queue)
    except BROKER_ERRORS:
        return None
    rows = [r for r in await _capabilities(queue) if r.host in hosts]
    uuids = sorted({u for r in rows for u in r.capabilities.get('gpu_uuids') or []})
    if not rows:
        return None
    async with async_session_factory()() as session, deployment_tx(session, 'trainer gpu order'):
        found = await session.execute(
            select(GpuDevice.nvml_index).where(GpuDevice.gpu_uuid.in_(uuids))
        )
        positions = list(found.scalars())
    order = parse_order(env_str('OP_TRAIN_GPU_ORDER'))
    return sorted(host_id_at(order, position) for position in positions)


async def probe_trainer_reachable() -> tuple[str, str]:
    """``(severity, detail)`` with severity ``ok``, ``warn`` or ``block``.

    No consumer on ``gpu_train`` blocks: the run would wait the whole queue-wait limit. A broker
    that cannot be asked, or a consumer whose capabilities row is missing, only warns.
    """
    try:
        hosts = await consumer_hosts()
    except BROKER_ERRORS as exc:
        return 'warn', f'cannot ask the broker for the trainer ({exc}); not verified'
    if not hosts:
        return 'block', 'trainer not running: no worker consumes the gpu_train queue'
    rows = [r for r in await _capabilities(TaskQueues.GPU_TRAIN) if r.host in hosts]
    if not rows:
        return 'warn', f'trainer on {sorted(hosts)} consumes gpu_train but recorded no capabilities'
    return 'ok', f'trainer available on {sorted(hosts)}'


__all__ = ['consumer_hosts', 'probe_trainer_reachable', 'read_trainer_gpu_order']

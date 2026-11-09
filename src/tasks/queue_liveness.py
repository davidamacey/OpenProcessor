"""A worker of a CPU-only queue says it is alive: ``worker_capabilities`` as a liveness table.

GPU workers write their ``worker_capabilities`` row from the card inventory
(``src/tasks/runtime/inventory.py``). A consumer of a queue with no card behind it (``cluster``,
``arbiter``) has none, yet the beat's ``cluster_refresh`` tick must know a consumer is there before
it queues a run (review RR-21: a fresh row, never a broadcast ``inspect()``), and the training
preflight must know the GPU arbiter worker is there before it promises a stop. When such a worker is ready, a
daemon thread upserts the row every ``worker_liveness.INTERVAL_S`` seconds; the reader accepts a
row younger than ``worker_liveness.FRESH_S``. The thread uses its own unpooled connection per
touch, so the prefork parent holds no pooled socket its children could inherit
(https://docs.celeryq.dev/en/stable/userguide/signals.html#worker-ready).
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any

from celery.signals import worker_ready, worker_shutting_down

from src.config.queue import TaskQueues


logger = logging.getLogger(__name__)

LIVENESS_QUEUES = (TaskQueues.CLUSTER, TaskQueues.ARBITER)
_stop = threading.Event()


def touch(queue: str) -> None:
    """Upsert this host's liveness row for ``queue`` on a one-off connection."""
    from sqlalchemy.orm import Session

    from src.db.engine import one_off_connection
    from src.db.repos import worker_liveness
    from src.db.session import sync_deployment_tx

    with (
        one_off_connection('app', f'op-liveness-{queue}', direct=False) as conn,
        Session(bind=conn) as session,
        sync_deployment_tx(session, 'queue liveness'),
    ):
        worker_liveness.touch(session, queue, socket.gethostname())


def _beat(queues: tuple[str, ...]) -> None:
    from sqlalchemy.exc import SQLAlchemyError

    from src.db.repos import worker_liveness

    while True:
        for queue in queues:
            try:
                touch(queue)
            except (SQLAlchemyError, OSError) as exc:  # a missed touch only ages the row
                logger.warning('queue_liveness_failed queue=%s error=%s', queue, exc)
        if _stop.wait(worker_liveness.INTERVAL_S):
            return


def consumed(sender: Any) -> tuple[str, ...]:
    """The gated queues this worker consumes (after ``-Q`` selected them)."""
    consume_from = sender.app.amqp.queues.consume_from
    return tuple(q for q in LIVENESS_QUEUES if q in consume_from)


def _on_worker_ready(sender: Any = None, **_: Any) -> None:
    queues = consumed(sender)
    if queues:
        _stop.clear()
        threading.Thread(target=_beat, args=(queues,), name='queue-liveness', daemon=True).start()


def _on_worker_shutting_down(**_: Any) -> None:
    _stop.set()


def install() -> None:
    worker_ready.connect(_on_worker_ready, weak=False)
    worker_shutting_down.connect(_on_worker_shutting_down, weak=False)

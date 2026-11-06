"""``op.infra.noop`` through the real broker and a real worker (the B2 sweep, step 2).

Needs the harness with Valkey and a ``jobs-worker`` running with task events on (``-E``), and
``OP_QUEUE_URL`` / ``OP_QUEUE_PASSWORD_FILE`` set for this process. The task is a plain task,
so success is seen on the Celery event stream, not in a result backend (there is none).
"""

from __future__ import annotations

import uuid

import pytest

from src.clients.task_queue import enqueue, producer_app


pytestmark = pytest.mark.live

WAIT_S = 60


def test_noop_is_delivered_to_a_worker_and_succeeds() -> None:
    app = producer_app()
    job_id = str(uuid.uuid4())
    seen: dict[str, dict] = {}

    def on_event(event: dict) -> None:
        seen[event['uuid']] = event

    with app.connection_for_read() as connection:
        receiver = app.events.Receiver(connection, handlers={'task-succeeded': on_event})
        enqueue('op.infra.noop', job_id=job_id)
        for _ in receiver.itercapture(limit=None, timeout=WAIT_S, wakeup=True):
            if job_id in seen:
                break
    assert job_id in seen, f'op.infra.noop did not succeed within {WAIT_S} s'

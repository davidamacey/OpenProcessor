"""The producer: one message per enqueue, routed by the table, broker errors as QueueUnavailable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.clients import task_queue
from src.clients.task_queue import QueueUnavailable, UnknownTaskNameError, enqueue
from src.config.queue import TaskQueues


if TYPE_CHECKING:
    from pathlib import Path

    from celery import Celery


@pytest.fixture
def queue_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    password = tmp_path / 'queue_password'
    password.write_text('test-password\n')
    monkeypatch.setenv('OP_QUEUE_URL', 'redis://valkey:6379/0')
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(password))


def _memory_producer(monkeypatch: pytest.MonkeyPatch) -> Celery:
    """The real producer configuration with the in-memory broker substituted for Redis."""
    app = task_queue._build_producer('memory://')
    app.conf.broker_transport_options = {}
    monkeypatch.setattr(task_queue, 'producer_app', lambda: app)
    return app


def _drain(app: Celery, queue: str) -> list[dict[str, object]]:
    messages = []
    with app.connection_for_read() as connection, connection.SimpleQueue(queue) as simple:
        while True:
            try:
                message = simple.get(block=False)
            except simple.Empty:
                break
            messages.append({'headers': message.headers, 'body': message.payload})
            message.ack()
    return messages


def test_one_message_with_task_id_equal_to_job_id_on_the_routed_queue(
    monkeypatch: pytest.MonkeyPatch, queue_env: None
) -> None:
    app = _memory_producer(monkeypatch)
    enqueue('op.infra.noop', job_id='job-1')
    sent = _drain(app, TaskQueues.JOBS)
    assert len(sent) == 1
    assert sent[0]['headers']['id'] == 'job-1'  # type: ignore[index]
    assert sent[0]['headers']['task'] == 'op.infra.noop'  # type: ignore[index]
    assert sent[0]['body'][0] == ['job-1']  # type: ignore[index]
    for other in TaskQueues.ALL:
        if other != TaskQueues.JOBS:
            assert _drain(app, other) == []


def test_an_unknown_task_name_raises_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch, queue_env: None
) -> None:
    app = _memory_producer(monkeypatch)
    with pytest.raises(UnknownTaskNameError):
        enqueue('op.no.such_task', job_id='job-2')
    assert _drain(app, TaskQueues.JOBS) == []


def test_a_broker_error_raises_queue_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    password = tmp_path / 'pw'
    password.write_text('x\n')
    monkeypatch.setenv('OP_QUEUE_URL', 'redis://127.0.0.1:1/0')
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(password))
    task_queue.producer_app.cache_clear()
    with pytest.raises(QueueUnavailable):
        enqueue('op.infra.noop', job_id='job-3')


def test_the_producer_and_the_worker_carry_equal_transport_options(queue_env: None) -> None:
    from src.tasks.runtime import gpu_app

    task_queue.producer_app.cache_clear()
    assert (
        task_queue.producer_app().conf.broker_transport_options
        == gpu_app.make_app([]).conf.broker_transport_options
    )

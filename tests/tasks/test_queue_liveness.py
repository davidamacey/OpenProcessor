"""A worker that consumes the ``cluster`` queue keeps a fresh ``worker_capabilities`` row, so the
beat's ``cluster_refresh`` tick can tell a consumer is there. Other queues get no row from it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fixtures.job_rows import sql

from src.tasks import queue_liveness


def worker(*queues: str) -> SimpleNamespace:
    amqp = SimpleNamespace(queues=SimpleNamespace(consume_from={q: object() for q in queues}))
    return SimpleNamespace(app=SimpleNamespace(amqp=amqp))


def test_only_the_gated_queue_a_worker_consumes_is_kept_alive() -> None:
    assert queue_liveness.consumed(worker('cluster', 'jobs')) == ('cluster',)
    assert queue_liveness.consumed(worker('jobs', 'periodic')) == ()


def test_a_ready_worker_of_another_queue_starts_no_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[object] = []

    def record(thread: object) -> None:
        started.append(thread)

    monkeypatch.setattr(queue_liveness.threading.Thread, 'start', record)
    queue_liveness._on_worker_ready(sender=worker('jobs'))
    assert started == []
    queue_liveness._on_worker_ready(sender=worker('cluster'))
    assert len(started) == 1


@pytest.mark.db
@pytest.mark.usefixtures('pg_env')
def test_a_touch_writes_a_fresh_row_and_a_repeat_only_refreshes_it(pg) -> None:
    sql(pg, 'DELETE FROM worker_capabilities')
    queue_liveness.touch('cluster')
    queue_liveness.touch('cluster')
    [(queue, fresh)] = sql(
        pg, "SELECT queue, seen_at > now() - interval '5 seconds' FROM worker_capabilities"
    )
    assert (queue, fresh) == ('cluster', True)

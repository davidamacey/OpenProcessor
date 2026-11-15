"""Who starts the ExifTool processes, and when: before the process opens a gRPC channel.

The API's lifespan and the pool children of a job worker (a Celery ``worker_process_init``) start
the pool; a read never has to."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from src.app import lifespan as lifespan_module
from src.services.curation import capture_read


@pytest.fixture(autouse=True)
def _no_pool_left() -> Any:
    capture_read.shutdown_capture_pool()
    yield
    capture_read.shutdown_capture_pool()


def test_the_pool_starts_every_process_and_the_threads_use_them() -> None:
    capture_read.start_capture_pool()
    started = capture_read.live_processes()
    assert len(started) == capture_read.pool_size(__import__('os').cpu_count() or 1)
    pool = capture_read._pool()
    seen: set[int] = set()
    lock = threading.Lock()
    barrier = threading.Barrier(pool.size)

    def work() -> None:
        barrier.wait(10)
        client = pool._client()
        with lock:
            seen.add(id(client))

    futures = [pool.executor.submit(work) for _ in range(pool.size)]
    for f in futures:
        f.result(15)
    assert len(seen) == pool.size
    assert sorted(capture_read.live_processes()) == sorted(started)  # nobody started a second set


@pytest.mark.asyncio
async def test_the_lifespan_starts_the_pool_before_any_channel_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI

    order: list[str] = []
    live_at: dict[str, int] = {}

    def step(name: str) -> Any:
        async def run(*_a: Any, **_k: Any) -> None:
            order.append(name)
            live_at[name] = len(capture_read.live_processes())

        return run

    monkeypatch.setattr(lifespan_module, '_preflight', lambda: order.append('preflight'))
    monkeypatch.setattr(
        lifespan_module.startup_gate, 'require_control_plane', step('control_plane')
    )
    monkeypatch.setattr(lifespan_module, '_start_pools', step('pools'))
    monkeypatch.setattr(lifespan_module, '_start_background_tasks', step('background'))
    monkeypatch.setattr(lifespan_module, '_start_event_listener', step('events'))
    monkeypatch.setattr(lifespan_module, '_warm_pe_encoder', lambda _app: None)
    monkeypatch.setattr(lifespan_module, '_warm_routes', step('routes'))

    async def no_shutdown() -> None:
        return None

    monkeypatch.setattr(lifespan_module, '_run_shutdown', no_shutdown)
    async with lifespan_module.lifespan(FastAPI()):
        pass
    assert order[:3] == ['preflight', 'control_plane', 'pools']
    # the control-plane probe is the first thing that opens a gRPC channel
    assert live_at['control_plane'] > 0
    assert live_at['pools'] == live_at['control_plane']


@pytest.mark.parametrize(
    ('queue', 'starts'), [('jobs', True), ('interactive', False), ('cluster', False)]
)
def test_a_job_worker_child_starts_the_pool_only_where_ingest_tasks_run(
    monkeypatch: pytest.MonkeyPatch, queue: str, starts: bool
) -> None:
    from src.tasks import app as tasks_app

    started: list[bool] = []
    monkeypatch.setattr(capture_read, 'start_capture_pool', lambda: started.append(True))
    queues = tasks_app.app.amqp.queues
    monkeypatch.setattr(queues, '_consume_from', {queue: queues[queue]})
    tasks_app._on_worker_process_init()
    assert bool(started) is starts

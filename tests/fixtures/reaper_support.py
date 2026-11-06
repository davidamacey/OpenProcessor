"""Shared by the reaper tests: a fake broker, the Reaper under test, a clean control plane."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from fixtures.job_rows import sql

from src.clients import task_queue
from src.services.jobs.reaper import Reaper


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterator

    import pytest
    from fixtures.postgres import PgDatabase


class Broker:
    """A fake producer and marker: records dispatches, answers a chosen marker state."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, uuid.UUID]] = []
        self.marker = task_queue.QueuedState.ABSENT
        self.down = False
        self.lock = threading.Lock()

    def dispatch(self, kind: str, job_id: uuid.UUID) -> None:
        if self.down:
            raise task_queue.QueueUnavailable('broker down')
        with self.lock:
            self.sent.append((kind, job_id))

    def state(self, _job_id: str) -> task_queue.QueuedState:
        return self.marker


def environment(pg: PgDatabase, monkeypatch: pytest.MonkeyPatch) -> Iterator[PgDatabase]:
    """The database environment, the announcements muted, and an empty control plane."""
    monkeypatch.setattr('src.services.jobs.reaper.wake', lambda _project: None)
    # the reaper acts across projects: start every test from an empty control plane
    for table in (
        'search_projection_tasks',
        'stream_worker_claims',
        'worker_capabilities',
        'gpu_devices',
        'events',
        'training_runs',
        'jobs',
    ):
        sql(pg, f'DELETE FROM {table}')
    yield pg
    from src.db.engine import dispose_engines

    asyncio.run(dispose_engines())


def reaper(broker: Broker, **kwargs) -> Reaper:
    return Reaper(queued_state=broker.state, dispatcher=broker.dispatch, **kwargs)


def row(pg, job_id):
    cols = 'status, retry_count, attempt, error_code, owner, dispatch_count, pause_reason'
    [found] = sql(pg, f'SELECT {cols} FROM jobs WHERE id = %s', job_id)
    return found


def status(pg, job_id) -> str:
    return row(pg, job_id)[0]

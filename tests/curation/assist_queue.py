"""Plugin: the curation stack's assist runs through the real job queue wrapper.

``assist_queue`` gives the test a migrated database with the bound project's row (the VLM
and prompt configuration stay in the in-memory config repository, and published events
are captured, as the stack is built on them), a fake
broker, and a stand-in worker: every message the broker receives runs the ``assist_run``
handler in a thread through :func:`src.tasks.runtime.job_task.run_job` (the claim, the
heartbeat, the fenced terminal write), against the stack's fake item store. The heartbeat
ticks fast so a cancel reaches the running handler within a moment.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from curation.jobs_a.optin_queue import job_env, kinds, producer, wakes  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


def _run_queued(job_id: str, opensearch: Any) -> None:
    from src.config.job_kinds import KINDS
    from src.tasks import jobs_a
    from src.tasks.runtime.job_task import run_job

    async def opensearch_factory() -> Any:
        return opensearch

    run_job(KINDS['assist_run'], jobs_a.assist_run, {'opensearch': opensearch_factory}, job_id)


@pytest.fixture
def assist_queue(
    job_env: Any,  # noqa: F811 - the fixture is requested by name
    memory_config: Any,
    captured_events: Any,
    producer: Any,  # noqa: F811 - the fixture is requested by name
    wakes: Any,  # noqa: F811 - the fixture is requested by name
    stack: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[threading.Thread]]:
    from _records_world import make_project_row

    from src.config.project_context import bind_project, current_project
    from src.domain import job_timing
    from src.tasks import jobs_a_support

    project = make_project_row(job_env, f'default-{uuid.uuid4().hex[:8]}')
    monkeypatch.setattr(current_project().record, 'id', project, raising=False)
    monkeypatch.setattr(job_timing, 'HEARTBEAT_TICK_S', 0.05)

    record = current_project().record

    @asynccontextmanager
    async def bound(_ctx: Any) -> AsyncIterator[None]:
        with bind_project(record):
            yield

    monkeypatch.setattr(jobs_a_support, 'bound_project', bound)
    threads: list[threading.Thread] = []

    def on_send(_task: str, job_id: str) -> None:
        thread = threading.Thread(target=_run_queued, args=(job_id, stack.items), daemon=True)
        thread.start()
        threads.append(thread)

    producer.on_send = on_send
    yield threads
    for thread in threads:
        thread.join(timeout=30)

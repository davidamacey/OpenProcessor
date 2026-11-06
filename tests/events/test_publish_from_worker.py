"""A publish from a Celery-task context (no FastAPI app, no event loop) reaches an API-side subscriber."""

from __future__ import annotations

import asyncio

import pytest
from _events_support import seed_project
from _listener_support import running_listener, take

from src.services.events.publish import publish


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.mark.asyncio
async def test_a_worker_thread_publish_reaches_the_api_subscriber(pg_env, broker) -> None:
    seed_project(pg_env, 'worker-pub')
    async with running_listener(broker) as listener:
        sub = await listener.subscribe('worker-pub')
        # A pool child has no running loop in the publishing thread: it uses the sync engine.
        await asyncio.to_thread(publish, 'job.state', {'state': 'running'}, project='worker-pub')
        rows = await take(sub, 1)
    assert [(r.type, r.payload['state']) for r in rows] == [('job.state', 'running')]
    assert broker.published == [f'op:events:{sub.project_id}']

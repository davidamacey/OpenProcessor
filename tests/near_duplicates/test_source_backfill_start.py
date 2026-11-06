"""The start route of the source backfill: one job per project, a second start is busy."""

from __future__ import annotations

import pytest

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.routers.curation import source_backfill as route


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def test_a_start_queues_the_job_and_a_second_start_is_busy(
    project, job_env, producer, wakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(route, 'current_project_id', lambda: project)
    async with async_session_factory()() as session:
        started = await route.start_source_backfill(session)
    assert started.kind == 'source_backfill'
    assert producer.sent() == [started.job_id]
    with pytest.raises(RequestRejectedError) as busy:
        async with async_session_factory()() as session:
            await route.start_source_backfill(session)
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')

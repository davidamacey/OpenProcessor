"""The start route of the image-counts backfill: one job per project, a second start is busy."""

from __future__ import annotations

import pytest

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.routers.curation import image_counts as route


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def test_a_start_queues_the_job_and_a_second_start_is_busy_naming_the_running_job(
    project, job_env, producer, wakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(route, 'current_project_id', lambda: project)
    async with async_session_factory()() as session:
        started = await route.start_image_counts_backfill(session)
    assert started.kind == 'image_counts_backfill'
    assert producer.sent() == [started.job_id]
    with pytest.raises(RequestRejectedError) as busy:
        async with async_session_factory()() as session:
            await route.start_image_counts_backfill(session)
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')
    assert started.job_id in str(busy.value.detail)

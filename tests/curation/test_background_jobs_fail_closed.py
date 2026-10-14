"""A background job that crashed is logged at shutdown, and the ingest dispatch
tick does its file I/O off the event loop."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from structlog.testing import capture_logs

from src.services.curation.background_jobs import stop_background_jobs
from src.services.curation.bulk_ingest import dispatcher


if TYPE_CHECKING:
    import pytest


def test_a_crashed_background_job_is_logged_at_shutdown() -> None:
    async def crash() -> None:
        raise RuntimeError('sweeper blew up')

    async def go() -> list[dict]:
        task = asyncio.create_task(crash())
        await asyncio.sleep(0)
        with capture_logs() as logs:
            await stop_background_jobs([task])
        return logs

    logs = asyncio.run(go())
    crashed = [entry for entry in logs if entry['event'] == 'background_job_crashed']
    assert len(crashed) == 1
    assert 'sweeper blew up' in crashed[0]['error']


def test_a_cancelled_background_job_is_not_a_crash() -> None:
    async def forever() -> None:
        await asyncio.sleep(3600)

    async def go() -> list[dict]:
        task = asyncio.create_task(forever())
        await asyncio.sleep(0)
        with capture_logs() as logs:
            await stop_background_jobs([task])
        return logs

    assert asyncio.run(go()) == []


def test_the_dispatch_tick_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    monkeypatch.setattr(
        'src.services.projects.bootstrap.for_each_project', lambda: iter(['alpha', 'beta'])
    )
    monkeypatch.setattr(
        dispatcher, 'run_project_tick', lambda: seen.append(threading.get_ident()) or []
    )

    async def go() -> int:
        await dispatcher.run_tick()
        return threading.get_ident()

    loop_thread = asyncio.run(go())
    assert len(seen) == 2
    assert loop_thread not in seen

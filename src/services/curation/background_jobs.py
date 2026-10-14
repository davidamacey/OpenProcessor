"""Periodic curation jobs the API starts in its lifespan and cancels at shutdown."""

from __future__ import annotations

import asyncio
import contextlib

from src.services.curation.bulk_ingest.dispatcher import start_ingest_dispatcher
from src.services.curation.open_vocab_sweeper import start_open_vocab_sweeper


def start_background_jobs() -> list[asyncio.Task[None]]:
    """Resume ingest-time open-vocabulary passes a restart left pending, and act on due
    source triggers (watch polls, cron schedules, webhook-queued objects). A job whose
    interval setting is 0 is not started."""
    started = (start_open_vocab_sweeper(), start_ingest_dispatcher())
    return [task for task in started if task is not None]


async def stop_background_jobs(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

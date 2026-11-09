"""Periodic curation jobs the API starts in its lifespan and cancels at shutdown."""

from __future__ import annotations

import asyncio

from src.core.logging import get_logger
from src.services.curation.bulk_ingest.dispatcher import start_ingest_dispatcher


logger = get_logger(__name__)


def start_background_jobs() -> list[asyncio.Task[None]]:
    """Act on due source triggers (watch polls, cron schedules, webhook-queued objects). A job
    whose interval setting is 0 is not started."""
    started = (start_ingest_dispatcher(),)
    return [task for task in started if task is not None]


async def stop_background_jobs(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        task.cancel()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    for task, outcome in zip(tasks, outcomes, strict=True):
        if isinstance(outcome, Exception):  # a CancelledError is the cancel above, not a crash
            logger.error('background_job_crashed', task=task.get_name(), error=str(outcome))

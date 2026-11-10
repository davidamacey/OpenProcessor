"""``op.periodic.region_text_consensus_sweep``: start a consensus once a burst of readings settled.

The reader sets the project's ``region_text_consensus_marker`` (``{"dirty": true}``) each time it
writes readings, which also moves the row's ``updated_at``. A marker still untouched for
``SETTLE_S`` means the burst is over, so this tick starts ``region_text_consensus``. The kind is
exclusive per project (group ``region_text_write``), so a start while a reread or a consensus is
live answers ``job_busy`` and waits for the next tick; the idempotency key carries the marker
revision, so a message delivered twice starts one job. A project with no marker, or one the last
consensus cleared, starts nothing.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

from src.config.advisory_locks import REGION_TEXT_SWEEP
from src.core.exceptions import RequestRejectedError
from src.db.repos.records import get_project_state
from src.db.session import async_session_factory, project_tx
from src.services.jobs.service import start_job
from src.services.periodic.runner import TickProject, TickReport, for_each_project


if TYPE_CHECKING:
    from datetime import datetime

KIND = 'region_text_consensus'
MARKER_KEY = 'region_text_consensus_marker'
SETTLE_S = 30
REQUESTER = 'beat'


def settled(value: object, updated_at: datetime, *, now: dt.datetime) -> bool:
    """``True`` when the marker says readings are waiting and none arrived in the last ``SETTLE_S``."""
    if not isinstance(value, dict) or value.get('dirty') is not True:
        return False
    return (now - updated_at) >= dt.timedelta(seconds=SETTLE_S)


async def tick(*, now: dt.datetime | None = None) -> TickReport:
    at = now or dt.datetime.now(dt.UTC)

    async def one(project: TickProject) -> None:
        async with async_session_factory()() as session, project_tx(session, project.id):
            marker = await get_project_state(session, project.id, MARKER_KEY)
        if marker is None or not settled(marker.value, marker.updated_at, now=at):
            return
        async with async_session_factory()() as session:
            try:
                await start_job(
                    session,
                    KIND,
                    project.id,
                    requested_by=REQUESTER,
                    idempotency_key=f'{KIND}:{project.id}:{marker.revision}',
                )
            except RequestRejectedError as exc:
                if isinstance(exc.detail, dict) and exc.detail.get('error') == 'job_busy':
                    return
                raise

    return await for_each_project(
        REGION_TEXT_SWEEP, one, task='op.periodic.region_text_consensus_sweep'
    )

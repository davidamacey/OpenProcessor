"""``op.periodic.ingest_dispatch``: act on every project's due ingest triggers.

One pass of :func:`dispatch_once` per active project, each under the project's ``INGEST_DISPATCH``
advisory lock (a project another tick is already working on is skipped). The decisions and the
job starts stay in ``bulk_ingest/dispatcher.py``; this is the loop around them that used to be an
asyncio task of the API process.
"""

from __future__ import annotations

from src.config.advisory_locks import INGEST_DISPATCH
from src.db.session import async_session_factory
from src.services.curation.bulk_ingest.dispatcher import dispatch_once
from src.services.periodic.runner import TickProject, TickReport, for_each_project


async def tick() -> TickReport:
    async def one(project: TickProject) -> None:
        async with async_session_factory()() as session:
            await dispatch_once(session, project.id)

    return await for_each_project(INGEST_DISPATCH, one, task='op.periodic.ingest_dispatch')

"""The queue rows of one source as the runner's ledger: claims and settlements in PostgreSQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.db.repos import ingest_sources as repo
from src.services.curation.bulk_ingest.runner import ClaimedPath
from src.services.curation.bulk_ingest.source_queue import TERMINAL_ERROR_KINDS, path_of


if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session


class DbLedger:
    """Implements ``runner.FileLedger`` over sessions in the job's project scope.

    ``session`` is ``JobContext.session``: it opens a session on the task's engine and commits
    on a clean exit, so each claim and each settlement is one short transaction."""

    def __init__(
        self,
        session: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        source_id: uuid.UUID,
        job_id: uuid.UUID,
    ) -> None:
        self._session = session
        self._source = source_id
        self._job = job_id

    async def remaining(self) -> int:
        async with self._session() as session:
            counts = await session.run_sync(lambda sync: repo.counts(sync, self._source))
        return counts['pending'] + counts['claimed'] + counts['failed']

    async def claim(self, limit: int) -> list[ClaimedPath]:
        async with self._session() as session:
            claimed = await session.run_sync(
                lambda sync: repo.claim(sync, self._source, self._job, limit)
            )
        return [ClaimedPath(c.file_key, path_of(c.file_key)) for c in claimed]

    async def settle(
        self,
        done: Sequence[tuple[str, str | None]],
        failed: Sequence[tuple[str, str, str]],
    ) -> None:
        failures = [
            (key, repo.error_text(error, kind), kind in TERMINAL_ERROR_KINDS)
            for key, error, kind in failed
        ]
        async with self._session() as session:

            def write(sync: Session) -> None:
                repo.mark_done(sync, self._source, self._job, list(done))
                repo.mark_failed(sync, self._source, self._job, failures)

            await session.run_sync(write)

    async def release(self) -> None:
        async with self._session() as session:
            await session.run_sync(lambda sync: repo.release_claims(sync, self._source, self._job))

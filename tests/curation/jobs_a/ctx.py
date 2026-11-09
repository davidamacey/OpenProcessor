"""A job context over the real database for handler tests: only the Celery wrapper is replaced.

``FakeCtx`` has the surface of ``src.tasks.runtime.context.JobContext`` that the family handlers
use. Its ``session()`` is a real project-scoped session, so ``job_items`` and records are real
rows; ``checkpoint`` and ``progress`` are recorded.
"""

from __future__ import annotations

import uuid  # noqa: TC003 - used in annotations of a plain class
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from src.db.session import async_session_factory, project_tx


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping


class FakeCtx:
    def __init__(
        self,
        project: uuid.UUID,
        job_id: uuid.UUID,
        params: Mapping[str, Any],
        *,
        resume_from: Any = None,
        cancel_after_checkpoints: int | None = None,
    ) -> None:
        self.project_id = project
        self.job_id = job_id
        self.params = params
        self.resume_from = resume_from
        self.project_slug = 'p'
        self.server = 'test-host'
        self.checkpoints: list[Any] = []
        self.progresses: list[tuple[Any, ...]] = []
        self._cancel_after = cancel_after_checkpoints
        self._cancel = False

    def cancelled(self) -> bool:
        return self._cancel

    def cancel_now(self) -> None:
        self._cancel = True

    async def checkpoint(self, cursor: Any) -> None:
        self.checkpoints.append(cursor)
        if self._cancel_after is not None and len(self.checkpoints) >= self._cancel_after:
            self._cancel = True

    async def progress(
        self,
        done: int,
        total: int | None = None,
        stage: str | None = None,
        counters: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.progresses.append((done, total, stage, dict(counters or {})))

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        async with async_session_factory()() as session, project_tx(session, self.project_id):
            yield session

    async def request_pause(self, reason: str) -> None:
        self.paused_for = reason

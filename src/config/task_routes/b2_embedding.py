"""Task routes of the `b2_embedding` research family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.jobs.embedding_reindex': TaskQueues.JOBS,
}

"""Task routes of the source backfill."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {'op.source_backfill.run': TaskQueues.JOBS}

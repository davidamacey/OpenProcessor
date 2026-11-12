"""Task routes of the per-image count backfill."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {'op.image_counts.backfill': TaskQueues.JOBS}

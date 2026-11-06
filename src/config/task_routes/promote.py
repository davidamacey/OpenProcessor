"""Task routes of the `promote` family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {'op.promote.run': TaskQueues.JOBS}

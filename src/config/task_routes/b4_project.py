"""Task routes of the `b4_project` family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {'op.b4_project.project_create': TaskQueues.JOBS}

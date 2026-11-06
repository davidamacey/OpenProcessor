"""Task routes of the `periodic` family: beat ticks, on queue `periodic`."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.periodic.reaper': TaskQueues.PERIODIC,
}

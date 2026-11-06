"""Task routes of the `families_b` family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.families_b.class_merge': TaskQueues.JOBS,
    'op.families_b.class_projection_sync': TaskQueues.JOBS,
}

"""Task routes of the near-duplicate service."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.near_dup.score': TaskQueues.JOBS,
    'op.near_dup.group': TaskQueues.JOBS,
    'op.near_dup.rescore': TaskQueues.JOBS,
    'op.near_dup.projection_sync': TaskQueues.JOBS,
}

"""Task routes of the core near-duplicate grouping."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.near_dup_core.group': TaskQueues.JOBS,
}

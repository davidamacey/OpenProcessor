"""Task routes of the clustering family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.clustering.run': TaskQueues.CLUSTER,
    'op.clustering.projection_sync': TaskQueues.JOBS,
}

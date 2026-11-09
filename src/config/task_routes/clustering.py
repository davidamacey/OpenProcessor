"""Task routes of the clustering family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.clustering.run': TaskQueues.CLUSTER,
    'op.clustering.projection_sync': TaskQueues.JOBS,
    'op.clustering.region_fp_centroids': TaskQueues.JOBS,
    'op.clustering.persons_cluster': TaskQueues.JOBS,
}

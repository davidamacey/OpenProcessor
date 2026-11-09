"""Task routes of the `periodic` family: beat ticks, on queue `periodic`."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.periodic.reaper': TaskQueues.PERIODIC,
    'op.periodic.ingest_dispatch': TaskQueues.PERIODIC,
    'op.periodic.open_vocab_sweep': TaskQueues.PERIODIC,
    'op.periodic.upload_sweep': TaskQueues.PERIODIC,
    'op.periodic.job_stats': TaskQueues.PERIODIC,
    'op.periodic.cache_prune': TaskQueues.PERIODIC,
    'op.periodic.reload_promoted': TaskQueues.PERIODIC,
    'op.periodic.serving_inventory': TaskQueues.PERIODIC,
    'op.periodic.resource_reachability': TaskQueues.PERIODIC,
    'op.periodic.cluster_refresh': TaskQueues.PERIODIC,
    'op.periodic.db_backup': TaskQueues.PERIODIC,
    # The one tick that runs in the process holding the Docker socket (gpu-arbiter-worker).
    'op.periodic.gpu_arbiter': TaskQueues.ARBITER,
    'op.periodic.near_dup_rescore': TaskQueues.PERIODIC,
    'op.periodic.near_dup_full_build': TaskQueues.PERIODIC,
}

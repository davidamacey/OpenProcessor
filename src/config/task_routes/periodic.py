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
    'op.periodic.near_dup_rescore': TaskQueues.PERIODIC,
    'op.periodic.near_dup_full_build': TaskQueues.PERIODIC,
}

"""The beat ticks (``op.periodic.*``), each a plain idempotent task on queue ``periodic``.

The bodies live in ``src/services/periodic/``; a task only starts one and returns. The limits are
the same for every tick (``soft_time_limit=120``, ``time_limit=150``): a tick is short work, and a
slow one is killed rather than overlapping the next.
"""

from __future__ import annotations

from src.services.periodic import (
    cache_prune,
    cluster_refresh,
    db_backup,
    gpu_arbiter,
    ingest_dispatch,
    job_stats,
    near_dup_ticks,
    open_vocab_sweep,
    reload_promoted,
    resource_reachability,
    serving_inventory,
    upload_sweep,
)
from src.services.periodic.runner import run_periodic
from src.tasks.app import app


_TICK = {'shared': False, 'soft_time_limit': 120, 'time_limit': 150, 'ignore_result': True}
# A dump of a large database outlasts a short tick; the limits sit above db_backup.DUMP_TIMEOUT_S.
_BACKUP_TICK = {**_TICK, 'soft_time_limit': 3300, 'time_limit': 3600}


@app.task(name='op.periodic.ingest_dispatch', **_TICK)
def ingest_dispatch_tick() -> None:
    """Start the poll, scan and drain jobs that are due, for every active project."""
    run_periodic('ingest_dispatch', ingest_dispatch.tick())


@app.task(name='op.periodic.open_vocab_sweep', **_TICK)
def open_vocab_sweep_tick() -> None:
    """Start the next open-vocabulary drain of each project that left images pending."""
    run_periodic('open_vocab_sweep', open_vocab_sweep.tick())


@app.task(name='op.periodic.upload_sweep', **_TICK)
def upload_sweep_tick() -> None:
    """Remove extracted dataset uploads past their time that no import refers to."""
    run_periodic('upload_sweep', upload_sweep.tick())


@app.task(name='op.periodic.job_stats', **_TICK)
def job_stats_tick() -> None:
    """Set the queue and job gauges from the database."""
    run_periodic('job_stats', job_stats.tick())


@app.task(name='op.periodic.near_dup_rescore', **_TICK)
def near_dup_rescore_tick() -> None:
    """Start the incremental near-duplicate rescore of each level with pending members."""
    run_periodic('near_dup_rescore', near_dup_ticks.rescore_tick())


@app.task(name='op.periodic.near_dup_full_build', **_TICK)
def near_dup_full_build_tick() -> None:
    """Start the weekly full near-duplicate build that bounds incremental drift."""
    run_periodic('near_dup_full_build', near_dup_ticks.full_build_tick())


@app.task(name='op.periodic.cache_prune', **_TICK)
def cache_prune_tick() -> None:
    """Evict the oldest crops once the tmpfs crop cache is over its cap."""
    run_periodic('cache_prune', cache_prune.tick())


@app.task(name='op.periodic.serving_inventory', **_TICK)
def serving_inventory_tick() -> None:
    """Record what each Triton server of the active serving profile reports."""
    run_periodic('serving_inventory', serving_inventory.tick())


@app.task(name='op.periodic.reload_promoted', **_TICK)
def reload_promoted_tick() -> None:
    """Re-load promoted models that Triton lost in a restart."""
    run_periodic('reload_promoted', reload_promoted.tick())


@app.task(name='op.periodic.resource_reachability', **_TICK)
def resource_reachability_tick() -> None:
    """Probe the monitoring services and publish the answer the API serves."""
    run_periodic('resource_reachability', resource_reachability.tick())


@app.task(name='op.periodic.cluster_refresh', **_TICK)
def cluster_refresh_tick() -> None:
    """Start a cluster run for each project whose item pool grew past its profile's rule."""
    run_periodic('cluster_refresh', cluster_refresh.tick())


@app.task(name='op.periodic.db_backup', **_BACKUP_TICK)
def db_backup_tick() -> None:
    """Dump the control-plane database to the backup directory and prune old dumps."""
    run_periodic('db_backup', db_backup.tick())


@app.task(name='op.periodic.gpu_arbiter', **_TICK)
def gpu_arbiter_tick() -> None:
    """Stop the GPU-resident services the running claims need and start the ones they freed."""
    run_periodic('gpu_arbiter', gpu_arbiter.tick())

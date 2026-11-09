"""The beat ticks (``op.periodic.*``), each a plain idempotent task on queue ``periodic``.

The bodies live in ``src/services/periodic/``; a task only starts one and returns. The limits are
the same for every tick (``soft_time_limit=120``, ``time_limit=150``): a tick is short work, and a
slow one is killed rather than overlapping the next.
"""

from __future__ import annotations

from src.services.periodic import ingest_dispatch, job_stats, open_vocab_sweep, upload_sweep
from src.services.periodic.runner import run_periodic
from src.tasks.app import app


_TICK = {'shared': False, 'soft_time_limit': 120, 'time_limit': 150, 'ignore_result': True}


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

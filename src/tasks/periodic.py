"""The beat ticks (``op.periodic.*``), each a plain idempotent task on queue ``periodic``.

The bodies live in ``src/services/periodic/``; a task only starts one and returns. The limits are
the same for every tick (``soft_time_limit=120``, ``time_limit=150``): a tick is short work, and a
slow one is killed rather than overlapping the next.
"""

from __future__ import annotations

from src.services.periodic import ingest_dispatch
from src.services.periodic.runner import run_periodic
from src.tasks.app import app


_TICK = {'shared': False, 'soft_time_limit': 120, 'time_limit': 150, 'ignore_result': True}


@app.task(name='op.periodic.ingest_dispatch', **_TICK)
def ingest_dispatch_tick() -> None:
    """Start the poll, scan and drain jobs that are due, for every active project."""
    run_periodic('ingest_dispatch', ingest_dispatch.tick())

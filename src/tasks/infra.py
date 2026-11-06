"""Infrastructure tasks: the no-op the live queue check sends."""

from __future__ import annotations

from src.tasks.app import app


@app.task(name='op.infra.noop', shared=False)
def noop(job_id: str) -> None:
    """Does nothing; proves a message reaches a worker through the routed queue."""

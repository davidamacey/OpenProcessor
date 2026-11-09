"""What the active bake-offs still score, for the retention prunes.

The start of a bake-off stores its job spec in the ``jobs.params`` of the ``bakeoff`` job, so
the prunes read the runs and datasets a queued, running, paused or cancelling bake-off scores
from PostgreSQL. The read is deployment-wide on purpose: a pin that belongs to another project
protects nothing of this one, and an unreadable database must stop the prune, not let it delete.
"""

from __future__ import annotations

from typing import Any

from src.db.repos import bakeoff as bakeoff_repo
from src.db.session import sync_deployment_tx, sync_session_factory


def active_bakeoff_specs() -> list[dict[str, Any]]:
    """The job spec (``BakeoffJobSpec`` as stored) of every active bake-off."""
    with (
        sync_session_factory()() as session,
        sync_deployment_tx(session, 'retention bake-off pins'),
    ):
        return [run.params for run in bakeoff_repo.active_bakeoffs(session)]

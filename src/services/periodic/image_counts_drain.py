"""``op.periodic.image_counts_drain``: recount the images whose items changed (R-13).

Item writers mark their image in a Valkey set (``src/clients/image_counts_dirty.py``); this tick
claims each active project's set and recomputes those images' stored counts from the items, then
recounts a bounded number of images that were never counted. It is the only writer of
``counts``/``counts_at``. A project's drain holds its advisory lock, so two ticks never recount the
same project at once; a tick that dies half way leaves the claimed set, which the next tick works
first.

The tick runs every project in one Celery task with ``soft_time_limit=120`` and ``time_limit=150``
(``src/tasks/periodic.py``). The soft limit raises ``SoftTimeLimitExceeded`` in the task and the
hard limit terminates the worker child, which cannot be caught
(https://docs.celeryq.dev/en/stable/userguide/workers.html#time-limits). So the drain time is a
budget of the whole tick, :data:`TICK_BUDGET_S`, split fairly: each project gets the time left
divided by the projects still to run. A project with a deep backlog stops at its share and the
claimed set carries the rest to the next tick, instead of starving the projects after it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Final

from src.clients.image_counts_dirty import DirtyImages, valkey_client
from src.config.advisory_locks import IMAGE_COUNTS_DRAIN
from src.services.curation.image_counts_refresh import DRAIN_BUDGET_S, drain
from src.services.curation.job_binding import opensearch_client
from src.services.periodic.runner import (
    TickProject,
    TickReport,
    for_each_project,
    read_tick_projects,
)


#: The drain time of one whole tick, below the task's 120 s soft limit with room for each
#: project's index refresh, sweep and client set-up.
TICK_BUDGET_S: Final = 60.0


async def drain_bound_project(budget_s: float = DRAIN_BUDGET_S) -> None:
    """Drain the bound project (its OpenSearch client and its dirty set open for the call)."""
    opensearch = await opensearch_client()
    valkey = valkey_client()
    try:
        await drain(opensearch, DirtyImages(valkey), budget_s=budget_s)
    finally:
        await valkey.aclose()
        await opensearch.close()


async def tick() -> TickReport:
    projects = await asyncio.to_thread(read_tick_projects)
    deadline = time.monotonic() + TICK_BUDGET_S
    left = len(projects)

    async def one(_project: TickProject) -> None:
        nonlocal left
        share = max(deadline - time.monotonic(), 0.0) / max(left, 1)
        left -= 1
        await drain_bound_project(share)

    return await for_each_project(IMAGE_COUNTS_DRAIN, one, task='op.periodic.image_counts_drain')

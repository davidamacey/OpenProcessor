"""``op.periodic.image_counts_drain``: recount the images whose items changed (R-13).

Item writers mark their image in a Valkey set (``src/clients/image_counts_dirty.py``); this tick
claims each active project's set and recomputes those images' stored counts from the items, then
recounts a bounded number of images that were never counted. It is the only writer of
``counts``/``counts_at``. A project's drain holds its advisory lock, so two ticks never recount the
same project at once; a tick that dies half way leaves the claimed set, which the next tick works
first.
"""

from __future__ import annotations

from src.clients.image_counts_dirty import DirtyImages, valkey_client
from src.config.advisory_locks import IMAGE_COUNTS_DRAIN
from src.services.curation.image_counts_refresh import drain
from src.services.curation.job_binding import opensearch_client
from src.services.periodic.runner import TickProject, TickReport, for_each_project


async def drain_bound_project() -> None:
    """Drain the bound project (its OpenSearch client and its dirty set open for the call)."""
    opensearch = await opensearch_client()
    valkey = valkey_client()
    try:
        await drain(opensearch, DirtyImages(valkey))
    finally:
        await valkey.aclose()
        await opensearch.close()


async def tick() -> TickReport:
    async def one(_project: TickProject) -> None:
        await drain_bound_project()

    return await for_each_project(IMAGE_COUNTS_DRAIN, one, task='op.periodic.image_counts_drain')

"""Tasks of the `b4_fanout` family: the batch routes' writes above ``OP_BATCH_SYNC_MAX``.

Each handler binds the job's project, takes the OpenSearch client the context injects (never
imported by ``src/tasks/runtime/``, review R-2) and runs the service's job body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.job_binding import bound_job_project, opensearch_client, project_record
from src.services.curation.new_class_resolve import run_resolve_job
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext

_FACTORIES = {'opensearch': opensearch_client}


async def review_resolve_batch(ctx: JobContext) -> dict[str, Any]:
    record = await project_record(ctx)
    with bound_job_project(ctx, record):
        return await run_resolve_job(ctx, await ctx.client('opensearch'))


register_job('review_resolve_batch', review_resolve_batch, client_factories=_FACTORIES)

"""Tasks of lane 4A: each handler binds the job's project, builds its clients and runs the
family's existing coroutine (03d section 1, recipe step 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.tasks import jobs_a_support as support
from src.tasks.runtime.context import JobResult
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext


async def _opensearch() -> Any:
    return await support.opensearch_client()


async def _ingest() -> Any:
    return await support.worker_ingest()


FACTORIES = {'opensearch': _opensearch, 'ingest': _ingest}


async def reprocess(ctx: JobContext) -> JobResult:
    from src.services.curation.reprocess_job import run_reprocess

    async with support.bound_project(ctx):
        opensearch = await ctx.client('opensearch')
        service = await (await ctx.client('ingest')).service(opensearch)
        result, with_errors = await run_reprocess(ctx, opensearch, service)
    return JobResult(result, with_errors=with_errors)


async def probe(ctx: JobContext) -> JobResult:
    from src.services.curation.probe_job import run_probe

    async with support.bound_project(ctx):
        opensearch = await ctx.client('opensearch')
        return JobResult(await run_probe(ctx, opensearch))


async def item_scores(ctx: JobContext) -> JobResult:
    from src.services.curation.item_scores.job import run_scoring_job

    async with support.bound_project(ctx):
        opensearch = await ctx.client('opensearch')
        return JobResult(await run_scoring_job(ctx, opensearch, list(ctx.params['scorers'])))


async def selection(ctx: JobContext) -> JobResult:
    from src.services.curation.selection.job import run_selection_job

    async with support.bound_project(ctx):
        return JobResult(await run_selection_job(ctx, await ctx.client('opensearch')))


async def embedding_viz(ctx: JobContext) -> JobResult:
    from src.services.curation.embedding_viz import run_projection_job

    async with support.bound_project(ctx):
        return JobResult(await run_projection_job(ctx, await ctx.client('opensearch')))


register_job('reprocess', reprocess, client_factories=FACTORIES)
OPENSEARCH_ONLY = {'opensearch': _opensearch}
register_job('probe', probe, client_factories=OPENSEARCH_ONLY)
register_job('item_scores', item_scores, client_factories=OPENSEARCH_ONLY)
register_job('selection', selection, client_factories=OPENSEARCH_ONLY)
register_job('embedding_viz', embedding_viz, client_factories=OPENSEARCH_ONLY)

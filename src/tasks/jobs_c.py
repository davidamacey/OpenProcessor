"""Tasks of lane 4C: project delete, ingest and the open-vocabulary drain.

Each handler calls the family's existing coroutine with clients from the job context, never from
a request (stage 4 recipe step 3). The OpenSearch factory is injected here and not in
``src/tasks/runtime/`` (review R-2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from src.tasks.runtime.context import JobContext

OPENSEARCH_TIMEOUT_S = 60


async def _opensearch() -> Any:
    """A project-guarded client of this task's own; the wrapper closes it with the task."""
    from src.config.settings import get_settings
    from src.services.projects.guard import make_script_opensearch

    return make_script_opensearch([get_settings().opensearch_url], timeout=OPENSEARCH_TIMEOUT_S)


async def bound_project(ctx: JobContext) -> AbstractContextManager[Any]:
    """Bind the job's project for the handler (a worker has no request to bind it)."""
    from src.config.project_context import bind_project
    from src.services.projects.registry import ProjectRegistry

    registry = ProjectRegistry()
    await registry.ensure_fresh()
    record = registry.get(str(ctx.project_slug))
    if record is None:
        raise LookupError(f'no project {ctx.project_slug!r} for job {ctx.job_id}')
    return bind_project(record)


async def project_delete(ctx: JobContext) -> dict[str, Any]:
    """Finish the delete of ``params['slug']``; the rows go in batches with progress in the job."""
    from src.services.events.publish import apublish_global_event
    from src.services.projects.delete import delete_project_finish

    slug = str(ctx.params['slug'])

    async def rows_deleted(table: str, deleted: int) -> None:
        await ctx.checkpoint({'table': table, 'deleted': deleted})
        await ctx.progress(0, None, 'rows', {'table': table, 'deleted': deleted})

    finished = await delete_project_finish(
        await ctx.client('opensearch'), slug=slug, on_rows_deleted=rows_deleted
    )
    cleanup = finished.mlflow_cleanup.to_wire()
    await apublish_global_event(
        'project.deleted',
        target=slug,
        status=finished.record.status,
        revision=finished.record.revision,
        mlflow_cleanup=cleanup,
    )
    return {'project': slug, 'mlflow_cleanup': cleanup}


async def open_vocab_drain(ctx: JobContext) -> dict[str, Any]:
    """Run the active open-vocabulary set over every ``pending`` image of the project."""
    from src.services.classes.registry import get_class_registry
    from src.services.curation.ingest_service_factory import get_ingest_service
    from src.services.curation.open_vocab_ingest import drain_pending
    from src.services.detection.segmenter_http import segment_image_http

    with await bound_project(ctx):
        opensearch = await ctx.client('opensearch')
        service = await get_ingest_service(opensearch, get_class_registry())

        async def batch_done(tried: int) -> None:
            await ctx.progress(tried, None, 'drain')

        return await drain_pending(
            opensearch, service, segment_image_http, should_stop=ctx.cancelled, on_batch=batch_done
        )


register_job('project_delete', project_delete, client_factories={'opensearch': _opensearch})
register_job('open_vocab_drain', open_vocab_drain, client_factories={'opensearch': _opensearch})

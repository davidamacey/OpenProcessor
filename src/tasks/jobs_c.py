"""Tasks of lane 4C: project delete, ingest and the open-vocabulary drain.

Each handler calls the family's existing coroutine with clients from the job context, never from
a request (stage 4 recipe step 3). The OpenSearch factory is injected here and not in
``src/tasks/runtime/`` (review R-2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext

OPENSEARCH_TIMEOUT_S = 60


async def _opensearch() -> Any:
    """A project-guarded client of this task's own; the wrapper closes it with the task."""
    from src.config.settings import get_settings
    from src.services.projects.guard import make_script_opensearch

    return make_script_opensearch([get_settings().opensearch_url], timeout=OPENSEARCH_TIMEOUT_S)


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


register_job('project_delete', project_delete, client_factories={'opensearch': _opensearch})

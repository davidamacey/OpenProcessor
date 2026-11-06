"""Tasks of the `families_b` family: class registry writes, dataset import, combine.

Each handler binds the job's project for the length of the run (the run coroutines read their
index names and directories through the bound project, never through a constant), builds its
OpenSearch client from the factory registered here (the runtime never imports a client, R-2)
and calls the existing run coroutine with it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from src.tasks.runtime.context import JobResult
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.tasks.runtime.context import JobContext


async def _opensearch() -> Any:
    from src.config.settings import get_settings
    from src.services.projects.guard import make_script_opensearch

    settings = get_settings()
    return make_script_opensearch([settings.opensearch_url], timeout=settings.opensearch_timeout)


class ProjectUnavailableError(RuntimeError):
    """The job's project is not in the registry (fail closed: nothing runs unbound)."""


@contextmanager
def _bound(record: Any) -> Iterator[None]:
    from src.config.project_context import bind_project

    with bind_project(record):
        yield


async def _record(ctx: JobContext) -> Any:
    from src.services.projects.registry import get_project_registry

    record = await get_project_registry().lookup(str(ctx.project_slug))
    if record is None:
        raise ProjectUnavailableError(f'project {ctx.project_slug!r} is not registered')
    return record


# --- class registry writes -------------------------------------------------------------------


async def class_merge(ctx: JobContext) -> Any:
    from src.services.classes.projection_sync import run_class_merge

    with _bound(await _record(ctx)):
        outcome = await run_class_merge(ctx, await ctx.client('opensearch'))
    return JobResult(outcome, with_errors=bool(outcome.get('errors')))


async def class_projection_sync(ctx: JobContext) -> Any:
    from src.services.classes.projection_sync import run_projection_sync

    with _bound(await _record(ctx)):
        return await run_projection_sync(ctx, await ctx.client('opensearch'))


_FACTORIES = {'opensearch': _opensearch}
register_job('class_merge', class_merge, client_factories=_FACTORIES)
register_job('class_projection_sync', class_projection_sync, client_factories=_FACTORIES)

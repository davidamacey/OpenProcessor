"""Handler of ``region_fp_centroids``: rebuild the false-positive centroids, then optionally pull
tight matches into the false-positive state.

Runs in a worker with the project's scope bound by the job wrapper; it binds the project's record
for the index names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.services.clustering.region_fp import (
    auto_assign_fp_from_centroids,
    build_region_fp_centroids,
)
from src.services.clustering.run import RunRefused
from src.services.projects.script_binding import resolve_project


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext


async def region_fp_centroids(ctx: JobContext) -> dict[str, Any]:
    if ctx.project_slug is None:
        raise RunRefused('region_fp_centroids needs a project')
    threshold = float(ctx.params.get('auto_fp_threshold') or 0.0)
    record = await resolve_project(ctx.project_slug)
    client = await ctx.client('opensearch')
    with bind_project(record):
        await ctx.progress(0, None, 'build')
        result: dict[str, Any] = {'fp_centroids': await build_region_fp_centroids(client)}
        if threshold > 0 and not ctx.cancelled():
            await ctx.progress(0, None, 'auto_fp')
            result['auto_fp'] = await auto_assign_fp_from_centroids(client, threshold=threshold)
    return {'status': 'success', **result}

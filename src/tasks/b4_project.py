"""Tasks of the `b4_project` module: project creation (the `project_create` job).

The route claims the slug (a ``building`` row) and queues this job; the handler builds the
directories and indexes and ends the row ``active`` (``failed`` with the reason on an error).
Every step checks what exists, so a re-run after a crash or a reaper resume finishes the same
project.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from src.services.curation.job_binding import opensearch_client
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext

OPENSEARCH_TIMEOUT_S = 60

_opensearch = functools.partial(opensearch_client, OPENSEARCH_TIMEOUT_S)


async def project_create(ctx: JobContext) -> dict[str, Any]:
    """Finish the create of ``params['slug']``; the result carries the warnings and stage seconds."""
    from src.services.events.publish import apublish_global_event
    from src.services.projects.create_stages import CreateStages
    from src.services.projects.lifecycle import build_project
    from src.services.projects.records import record_from_row
    from src.services.projects.store import get_row

    slug = str(ctx.params['slug'])
    row = await get_row(slug)
    record = record_from_row(row)
    if record.status == 'active':
        return {'project': slug, 'warnings': [], 'stages': {}}
    if record.status != 'building':
        raise RuntimeError(f"'{slug}' is {record.status}; delete it to free the slug")
    stages = CreateStages()
    await ctx.progress(0, 1, 'building')
    warnings = await build_project(
        await ctx.client('opensearch'),
        record,
        clone_settings_from=ctx.params.get('clone_settings_from'),
        clone_axes=ctx.params.get('clone_axes'),
        stages=stages,
    )
    await ctx.progress(1, 1, 'active')
    created = record_from_row(await get_row(slug))
    await apublish_global_event(
        'project.created', target=slug, status=created.status, revision=created.revision
    )
    return {'project': slug, 'warnings': warnings, 'stages': stages.seconds}


register_job('project_create', project_create, client_factories={'opensearch': _opensearch})

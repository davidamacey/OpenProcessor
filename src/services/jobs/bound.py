"""Bind the project a job runs for, for the body of its handler.

The OpenSearch guard, the index names and the embedding-space resolver all read the bound
project, so a handler that touches any of them runs inside this context. A job with no project,
or one the registry does not know, refuses (fail closed).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from src.config.project_context import bind_project
from src.services.projects.registry import get_project_registry


if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator

    from src.tasks.runtime.context import JobContext


class JobProjectError(RuntimeError):
    """The job has no project, or names one the registry does not know."""


@asynccontextmanager
async def bound_job_project(ctx: JobContext) -> AsyncIterator[uuid.UUID]:
    """Yield the job's project id with that project bound."""
    if ctx.project_id is None or ctx.project_slug is None:
        msg = f'job {ctx.job_id} has no project'
        raise JobProjectError(msg)
    registry = get_project_registry()
    await registry.ensure_fresh()
    record = registry.get(ctx.project_slug)
    if record is None:
        msg = f'project {ctx.project_slug!r} is not in the registry'
        raise JobProjectError(msg)
    with bind_project(record):
        yield ctx.project_id

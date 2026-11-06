"""What a curation job handler needs around its body: the bound project and the OpenSearch client.

A handler runs in a worker with no request, so nothing is bound. ``bound_job_project`` binds the
job's project for the block (and refuses a project that is gone or not bindable, fail closed) and
routes VLM usage rows through the job's own session. ``opensearch_client`` is the guarded client
factory the image registers for ``ctx.client('opensearch')`` (the runtime package never imports
an OpenSearch client).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.services.labeling.vlm_usage import usage_sessions
from src.services.projects.guard import make_script_opensearch
from src.services.projects.registry import ProjectRegistry


if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.config.projects import ProjectRecord
    from src.tasks.runtime.context import JobContext

_BINDABLE = frozenset({'active', 'archived'})


async def project_record(ctx: JobContext) -> ProjectRecord:
    """The registry record of the job's project; raises when it cannot be bound."""
    if ctx.project_slug is None:
        raise RuntimeError(f'job {ctx.job_id} ({ctx.kind}) has no project to bind')
    record = await ProjectRegistry().lookup(ctx.project_slug)
    if record is None or record.status not in _BINDABLE:
        status = None if record is None else record.status
        raise RuntimeError(f"project '{ctx.project_slug}' cannot be bound (status {status})")
    return record


@contextmanager
def bound_job_project(ctx: JobContext, record: ProjectRecord) -> Iterator[None]:
    with bind_project(record, read_only=record.status == 'archived'), usage_sessions(ctx.session):
        yield


async def opensearch_client() -> Any:
    from src.config.settings import get_settings

    settings = get_settings()
    return make_script_opensearch([settings.opensearch_url], timeout=settings.opensearch_timeout)

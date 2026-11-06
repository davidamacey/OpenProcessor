"""``register_job(kind, handler)``: turn a handler into the Celery task of a job kind.

The queue comes from the route table (never a decorator argument, which would override it), the
time limits and auto-resume from the kind table. The task's only argument is the job id; the
wrapper in ``job_task.py`` claims the row, heartbeats, runs the handler and writes the terminal
state. Tasks are registered with ``shared_task`` so the runtime never imports an app module:
every app that finalizes picks them up (Celery docs, "Creating tasks", ``shared_task``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from celery import shared_task

from src.config.job_kinds import KINDS
from src.tasks.runtime import gpu_app
from src.tasks.runtime.inventory import DbInventoryWriter
from src.tasks.runtime.job_task import run_job


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.tasks.runtime.context import ClientFactory, JobContext

Handler = Callable[['JobContext'], Awaitable[Any]]
_writer_installed = False


def register_job(
    kind: str,
    handler: Handler,
    *,
    client_factories: Mapping[str, ClientFactory] | None = None,
) -> Any:
    """Register ``handler`` as the task of ``kind`` and return the Celery task.

    ``client_factories`` maps a name to an async factory the handler reaches through
    ``await ctx.client(name)``; the OpenSearch factory is injected here by the image's own handler
    module, because this package never imports an OpenSearch client (review R-2).
    """
    spec = KINDS[kind]
    _ensure_inventory_writer(spec.queue)
    run = partial(run_job, spec, handler, dict(client_factories or {}))

    def task(job_id: str) -> None:
        run(job_id)

    task.__name__ = f'run_{spec.name}'
    task.__qualname__ = task.__name__
    return shared_task(
        name=spec.task_name,
        soft_time_limit=spec.soft_limit_s,
        time_limit=spec.hard_limit_s,
        ignore_result=True,
    )(task)


def _ensure_inventory_writer(queue: str) -> None:
    """The first kind an image registers names its queue in ``worker_capabilities``."""
    global _writer_installed  # noqa: PLW0603 - one writer per process
    if not _writer_installed:
        gpu_app.set_inventory_writer(DbInventoryWriter(queue))
        _writer_installed = True

"""Carry the request id from an HTTP request into the log lines of a Celery task.

The three signals of the ``asgi-correlation-id`` README (Integration with Celery): the id rides in a
message header on publish, the worker sets it before the task runs and clears it after. On top of
that the worker binds ``job_id`` (the task id, which is the job id) and ``kind`` (the task name)
with structlog's context variables, so every worker line carries them.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import structlog
from asgi_correlation_id import correlation_id
from celery.signals import before_task_publish, task_postrun, task_prerun


HEADER = 'op_request_id'


def transfer_request_id(headers: dict[str, Any], **_: Any) -> None:
    """``before_task_publish``: copy the current request id into the message headers."""
    request_id = correlation_id.get()
    if request_id:
        headers[HEADER] = request_id


def restore_request_id(task_id: str, task: Any, **_: Any) -> None:
    """``task_prerun``: the id from the header, else the one already in this context, else new.

    A task run in the same context as its caller (eager mode, a direct call) keeps the caller's id.
    """
    request_id = task.request.get(HEADER) or correlation_id.get() or uuid4().hex
    correlation_id.set(request_id)
    structlog.contextvars.bind_contextvars(job_id=task_id, kind=task.name)


def clear_request_id(**_: Any) -> None:
    """``task_postrun``: nothing from this task leaks into the next one in the same child."""
    correlation_id.set(None)
    structlog.contextvars.clear_contextvars()


def install_log_context() -> None:
    """Connect the three handlers once per process (``dispatch_uid`` dedupes repeated calls)."""
    before_task_publish.connect(
        transfer_request_id, dispatch_uid='op.log_context.publish', weak=False
    )
    task_prerun.connect(restore_request_id, dispatch_uid='op.log_context.prerun', weak=False)
    task_postrun.connect(clear_request_id, dispatch_uid='op.log_context.postrun', weak=False)

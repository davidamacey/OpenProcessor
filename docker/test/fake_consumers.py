"""Fake consumers for the queues whose real handlers need a GPU, a model or a Docker socket.

One module, one Celery app, four job kinds: each handler is the production worker shape
(``src/tasks/runtime``: claim, heartbeat, fenced progress, terminal state) with the work replaced
by a result that names the queue and the worker host, so a live check can prove which worker
claimed which job::

    celery -A fake_consumers:app worker -Q open_vocab    # also autolabel, gpu_export, gpu_eval

Docs: https://docs.celeryq.dev/en/stable/reference/cli.html#celery-worker (``-Q``).
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any

from src.config.job_kinds import KINDS
from src.tasks.runtime.gpu_app import make_app
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext

app = make_app(include=['fake_consumers'])

FAKE_KINDS = ('autolabel', 'open_vocab_drain', 'model_export', 'bakeoff')


def _handler(kind: str) -> Any:
    queue = KINDS[kind].queue

    async def fake(ctx: JobContext) -> dict[str, Any]:
        await ctx.progress(1, 1, 'fake')
        return {'fake': kind, 'queue': queue, 'host': socket.gethostname()}

    return fake


for _kind in FAKE_KINDS:
    register_job(_kind, _handler(_kind))

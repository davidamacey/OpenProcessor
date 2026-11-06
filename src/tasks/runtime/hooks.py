"""Seams the image's own modules fill, so ``src.tasks.runtime`` stays a leaf layer.

The runtime may import only ``src.db``, ``src.config``, ``src.domain`` and ``src.utils`` (the
trainer, export and evaluator images carry it without FastAPI, OpenSearch or torch). Two things
it needs live elsewhere:

* the producer side of the queue (``src.clients.task_queue``), used to send a yielded job's
  message again and to keep the ``op:queued:<id>`` marker honest. The API image installs it from
  ``src/tasks/app.py``; an image that does not install it cannot run a kind that yields.
* the router's choice of Triton server for a ``gpu_affinity = 'any_card'`` kind (GPU research
  S-11). Until a router installs one, the handler gets ``None`` and uses its configured server.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class HookError(Exception):
    """A queue hook could not reach the broker; the runtime logs it and carries on."""


class QueueHooks(Protocol):
    """The three producer calls of ``src.clients.task_queue``, raising :class:`HookError`."""

    def enqueue(self, task_name: str, *, job_id: str) -> None: ...

    def mark_queued(self, job_id: str, *, ttl_s: int) -> None: ...

    def clear_queued(self, job_id: str) -> None: ...


ServerChooser = Callable[[str], str | None]
"""``kind name -> server name`` (or ``None``): the router's choice for one job."""

_queue_hooks: QueueHooks | None = None
_server_chooser: ServerChooser | None = None


def install_queue_hooks(hooks: QueueHooks | None) -> None:
    global _queue_hooks  # noqa: PLW0603 - one producer per process, installed at app import
    _queue_hooks = hooks


def queue_hooks() -> QueueHooks | None:
    return _queue_hooks


def install_server_chooser(chooser: ServerChooser | None) -> None:
    global _server_chooser  # noqa: PLW0603 - one router per process, installed at app import
    _server_chooser = chooser


def choose_server(kind: str) -> str | None:
    return None if _server_chooser is None else _server_chooser(kind)

"""The producer hooks of the API image: ``src.clients.task_queue`` behind the runtime's seam."""

from __future__ import annotations

from src.clients import task_queue
from src.tasks.runtime.hooks import HookError, install_queue_hooks


class ProducerHooks:
    """Maps the producer's ``QueueUnavailable`` to the runtime's ``HookError``."""

    def enqueue(self, task_name: str, *, job_id: str) -> None:
        try:
            task_queue.enqueue(task_name, job_id=job_id)
        except task_queue.QueueUnavailable as exc:
            raise HookError(str(exc)) from exc

    def mark_queued(self, job_id: str, *, ttl_s: int) -> None:
        try:
            task_queue.mark_queued(job_id, ttl_s=ttl_s)
        except task_queue.QueueUnavailable as exc:
            raise HookError(str(exc)) from exc

    def clear_queued(self, job_id: str) -> None:
        try:
            task_queue.clear_queued(job_id)
        except task_queue.QueueUnavailable as exc:
            raise HookError(str(exc)) from exc


def install() -> None:
    install_queue_hooks(ProducerHooks())

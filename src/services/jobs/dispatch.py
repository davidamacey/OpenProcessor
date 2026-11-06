"""Send a queued job's message and set the queued marker.

Always after the transaction that made the row ``queued`` has committed (trap T-9: a worker must
never receive an id it cannot read). The broker is the only thing that can refuse; the marker is
a hint, so a failed marker never fails a dispatch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.clients import task_queue
from src.config.job_kinds import KINDS


if TYPE_CHECKING:
    import uuid

logger = logging.getLogger(__name__)


def dispatch(kind: str, job_id: uuid.UUID) -> None:
    """``send_task`` by name with ``task_id = job_id``, then ``SET op:queued:<id>``.

    Raises ``task_queue.QueueUnavailable`` when the broker refuses the message.
    """
    spec = KINDS[kind]
    task_queue.enqueue(spec.task_name, job_id=str(job_id))
    try:
        task_queue.mark_queued(str(job_id), ttl_s=spec.max_queue_wait_s)
    except task_queue.QueueUnavailable:
        logger.warning('queued_marker_not_set job_id=%s', job_id)


def forget_marker(job_id: uuid.UUID) -> None:
    """``DEL op:queued:<id>`` for a row that left ``queued``; best effort."""
    try:
        task_queue.clear_queued(str(job_id))
    except task_queue.QueueUnavailable:
        logger.warning('queued_marker_not_cleared job_id=%s', job_id)

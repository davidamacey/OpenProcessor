"""Task routes of the `b4_fanout` family: every kind runs on the ``jobs`` queue."""

from __future__ import annotations

from src.config.job_kinds.b4_fanout import KINDS
from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {kind.task_name: TaskQueues.JOBS for kind in KINDS}

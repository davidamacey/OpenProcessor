"""Task routes of lane 4A: every kind runs on the ``jobs`` queue."""

from __future__ import annotations

from src.config.job_kinds.families_a import KINDS
from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {kind.task_name: TaskQueues.JOBS for kind in KINDS}

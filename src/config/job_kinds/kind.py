"""The ``JobKind`` record: one row of the kind table in the job plan (02b section 2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config.queue import TaskQueues


ProjectScope = Literal['project', 'deployment', 'target', 'either']
MAX_ATTEMPTS = 3  # D-20: every kind


@dataclass(frozen=True)
class JobKind:
    """Pure data. ``register_job`` and ``start_job`` read queue, limits and auto-resume here.

    ``exclusive_group`` / ``exclusive_scope``: at most one active job per (project, group,
    scope value); the scope names the ``params`` key whose value narrows the group, ``''``
    meaning the whole project. ``None`` means no exclusivity.
    """

    name: str
    task_name: str
    queue: str
    project_scope: ProjectScope
    soft_limit_s: int
    hard_limit_s: int
    max_queue_wait_s: int
    exclusive_group: str | None = None
    exclusive_scope: str = ''
    auto_resume: bool = False
    max_attempts: int = MAX_ATTEMPTS

    def __post_init__(self) -> None:
        if self.queue not in TaskQueues.ALL:
            raise ValueError(f'job kind {self.name}: unknown queue {self.queue!r}')
        if not 0 < self.soft_limit_s < self.hard_limit_s:
            raise ValueError(f'job kind {self.name}: need 0 < soft limit < hard limit')

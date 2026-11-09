"""Job kinds of the `b4_project` family: project creation (02b section 2.1, B4 4D, R-21)."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='project_create',
        task_name='op.b4_project.project_create',
        queue=TaskQueues.JOBS,
        project_scope='deployment',
        soft_limit_s=3600,
        hard_limit_s=3600 + 300,
        max_queue_wait_s=3600,
        exclusive_group='project_create',
        exclusive_scope='slug',
        auto_resume=True,
    ),
)

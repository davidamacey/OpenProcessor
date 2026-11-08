"""Job kinds of the `promote` family: a trained model going live in Triton (02b section 2.1)."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='promote',
        task_name='op.promote.run',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=3 * 3600,
        hard_limit_s=3 * 3600 + 900,
        max_queue_wait_s=24 * 3600,
        exclusive_group='promote',
        exclusive_scope='triton_name',
    ),
)

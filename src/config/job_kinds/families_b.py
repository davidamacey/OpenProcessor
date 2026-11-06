"""Job kinds of the `families_b` family: class registry writes, dataset import, combine."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


_H = 3600
_QUEUE_WAIT_S = 24 * _H

KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='class_merge',
        task_name='op.families_b.class_merge',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=3 * _H,
        hard_limit_s=3 * _H + 900,
        max_queue_wait_s=_QUEUE_WAIT_S,
        exclusive_group='class_registry_write',
    ),
    JobKind(
        name='class_projection_sync',
        task_name='op.families_b.class_projection_sync',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=3 * _H,
        hard_limit_s=3 * _H + 900,
        max_queue_wait_s=_QUEUE_WAIT_S,
        exclusive_group='class_registry_write',
        auto_resume=True,
    ),
)

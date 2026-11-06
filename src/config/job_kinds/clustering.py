"""Job kinds of the clustering family (research clustering plan 5.2; 02b section 2.1)."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


_HOUR = 3600

KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='cluster_run',
        task_name='op.clustering.run',
        queue=TaskQueues.CLUSTER,
        project_scope='project',
        soft_limit_s=6 * _HOUR,
        hard_limit_s=6 * _HOUR + 15 * 60,
        max_queue_wait_s=24 * _HOUR,
        exclusive_group='cluster_run',
        exclusive_scope='scope_key',
    ),
    JobKind(
        name='cluster_projection_sync',
        task_name='op.clustering.projection_sync',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=3 * _HOUR,
        hard_limit_s=3 * _HOUR + 15 * 60,
        max_queue_wait_s=24 * _HOUR,
        exclusive_group='cluster_projection',
        exclusive_scope='scope_key',
        auto_resume=True,
    ),
)

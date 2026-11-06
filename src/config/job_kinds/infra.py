"""Infrastructure kinds: the no-op job the live check runs."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='noop',
        task_name='op.infra.noop',
        queue=TaskQueues.JOBS,
        project_scope='deployment',
        soft_limit_s=60,
        hard_limit_s=90,
        max_queue_wait_s=3600,
    ),
    JobKind(
        name='sleep_noop',
        task_name='op.infra.sleep_noop',
        queue=TaskQueues.JOBS,
        project_scope='deployment',
        soft_limit_s=660,
        hard_limit_s=720,
        max_queue_wait_s=3600,
    ),
)

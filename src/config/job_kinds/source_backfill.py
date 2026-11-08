"""Job kinds of the source backfill (job plan 2.1; review RR-10): one pausable pass, sliced."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='source_backfill',
        task_name='op.source_backfill.run',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=30 * 60,
        hard_limit_s=35 * 60,
        max_queue_wait_s=24 * 3600,
        exclusive_group='source_backfill',
        auto_resume=True,
        pausable=True,
        slice_s=600,
    ),
)

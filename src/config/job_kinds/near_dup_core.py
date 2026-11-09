"""Job kind of the core near-duplicate grouping (research near-duplicates plan 5.2).

The core global index belongs to no project, so the job is deployment-wide. Its exclusive group
``near_dup_core`` allows one writer at a time (the unique ``uq_jobs_one_active`` index): core
ingest never groups in the request, it writes pending rows and this job groups them.
"""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


_SOFT_S = 3 * 3600

KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='near_dup_core_group',
        task_name='op.near_dup_core.group',
        queue=TaskQueues.JOBS,
        project_scope='deployment',
        soft_limit_s=_SOFT_S,
        hard_limit_s=_SOFT_S + 15 * 60,
        max_queue_wait_s=3600,
        exclusive_group='near_dup_core',
        exclusive_scope='',
        auto_resume=True,
    ),
)

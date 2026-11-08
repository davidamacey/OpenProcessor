"""Job kinds of the near-duplicate service (research near-duplicates plan 5.2, job plan 2.1).

Every grouping kind shares the exclusive group ``near_dup`` with the level as scope, so one
writer runs per project and level (the unique ``uq_jobs_one_active`` index). The projection
has its own group so a sync never blocks a rescore.
"""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


_SOFT_S = 3 * 3600
_HARD_S = _SOFT_S + 15 * 60
_DAY_S = 24 * 3600


def _kind(name: str, group: str, *, auto_resume: bool, max_queue_wait_s: int = _DAY_S) -> JobKind:
    return JobKind(
        name=name,
        task_name=f'op.near_dup.{name.removeprefix("near_dup_")}',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=_SOFT_S,
        hard_limit_s=_HARD_S,
        max_queue_wait_s=max_queue_wait_s,
        exclusive_group=group,
        exclusive_scope='level',
        auto_resume=auto_resume,
    )


KINDS: tuple[JobKind, ...] = (
    _kind('near_dup_score', 'near_dup', auto_resume=True),
    _kind('near_dup_group', 'near_dup', auto_resume=False),
    _kind('near_dup_rescore', 'near_dup', auto_resume=True, max_queue_wait_s=3600),
    _kind('near_dup_projection_sync', 'near_dup_projection', auto_resume=True),
)

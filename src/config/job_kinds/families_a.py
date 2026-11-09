"""Job kinds of lane 4A: reprocess, assist runs, scores, probe, selection, viz, label operations."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


_HOUR_S = 3600
_QUARTER_S = 900
_DAY_WAIT_S = 24 * _HOUR_S


def _own_group(name: str) -> JobKind:
    """A 3 h kind that is its own exclusive group, project-wide (02b section 2.1)."""
    return JobKind(
        name=name,
        task_name=f'op.jobs_a.{name}',
        queue=TaskQueues.JOBS,
        project_scope='project',
        max_queue_wait_s=_DAY_WAIT_S,
        exclusive_group=name,
        soft_limit_s=3 * _HOUR_S,
        hard_limit_s=3 * _HOUR_S + _QUARTER_S,
    )


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='reprocess',
        task_name='op.jobs_a.reprocess',
        queue=TaskQueues.JOBS,
        project_scope='project',
        max_queue_wait_s=_DAY_WAIT_S,
        exclusive_group='reprocess',
        auto_resume=True,
        soft_limit_s=24 * _HOUR_S,
        hard_limit_s=24 * _HOUR_S + _QUARTER_S,
    ),
    _own_group('assist_run'),
    _own_group('item_scores'),
    _own_group('probe'),
    _own_group('selection'),
    _own_group('embedding_viz'),
    JobKind(
        name='label_op',
        task_name='op.jobs_a.label_op',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=_HOUR_S,
        hard_limit_s=_HOUR_S + 300,
        max_queue_wait_s=_HOUR_S,
        exclusive_group='label_op',
        exclusive_scope='op_id',
    ),
    JobKind(
        name='label_undo',
        task_name='op.jobs_a.label_undo',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=_HOUR_S,
        hard_limit_s=_HOUR_S + 300,
        max_queue_wait_s=_HOUR_S,
        exclusive_group='label_undo',
    ),
)

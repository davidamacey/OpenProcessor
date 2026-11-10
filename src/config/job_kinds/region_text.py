"""Job kinds of the registration-plate text family (research plate text plan 5.3, job plan 2.1).

Both kinds, and ``identity_text_rekey`` when it lands, share the exclusive group
``region_text_write`` with the whole project as scope, so no two writers touch the same boxes.
The bulk re-read has its own queue, ``text_read``, so a long read never starves ``jobs``.
"""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


_HOUR_S = 3600
_QUARTER_S = 15 * 60
_DAY_S = 24 * _HOUR_S

GROUP = 'region_text_write'


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='region_text_reread',
        task_name='op.region_text.reread',
        queue=TaskQueues.TEXT_READ,
        project_scope='project',
        soft_limit_s=_DAY_S,
        hard_limit_s=_DAY_S + _QUARTER_S,
        max_queue_wait_s=_DAY_S,
        exclusive_group=GROUP,
        auto_resume=True,
    ),
    JobKind(
        name='region_text_consensus',
        task_name='op.region_text.consensus',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=3 * _HOUR_S,
        hard_limit_s=3 * _HOUR_S + _QUARTER_S,
        max_queue_wait_s=_DAY_S,
        exclusive_group=GROUP,
        auto_resume=True,
    ),
)

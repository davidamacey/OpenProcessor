"""Job kinds of the `trigger` family: the auto-label sweep, the assist loop and the VLM batch routes.

Values are the kind table of the jobs plan (02b section 2.1): ``autolabel`` is pausable between
stages and VLM chunks; ``assist_loop`` runs in slices of ``slice_s`` seconds, each a delivery with
its own 30 minute soft limit, and waits up to 24 hours between slices behind other projects.
"""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='autolabel',
        task_name='op.autolabel.run',
        queue=TaskQueues.AUTOLABEL,
        project_scope='project',
        soft_limit_s=3 * 3600,
        hard_limit_s=3 * 3600 + 15 * 60,
        max_queue_wait_s=3600,
        exclusive_group='autolabel',
        pausable=True,
    ),
    JobKind(
        name='assist_loop',
        task_name='op.assist_loop.run',
        queue=TaskQueues.ASSIST_LOOP,
        project_scope='project',
        soft_limit_s=30 * 60,
        hard_limit_s=35 * 60,
        max_queue_wait_s=24 * 3600,
        exclusive_group='assist_loop',
        auto_resume=True,
        pausable=True,
        slice_s=600,
    ),
    JobKind(
        name='vlm_label_batch',
        task_name='op.vlm.label_batch',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=3 * 3600,
        hard_limit_s=3 * 3600 + 15 * 60,
        max_queue_wait_s=3600,
    ),
    JobKind(
        name='vlm_verify_region_batch',
        task_name='op.vlm.verify_region_batch',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=3 * 3600,
        hard_limit_s=3 * 3600 + 15 * 60,
        max_queue_wait_s=3600,
    ),
)
